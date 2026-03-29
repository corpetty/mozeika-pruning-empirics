"""
Experiment 20: V-specific k threshold scan.

Exp 19 confirmed online basis updating doesn't help V compression — the failure
is structural, not drift-related. V has higher intrinsic variance in its tail
dimensions (basis overlap 0.702 vs K's 0.825).

This experiment finds the minimum k_V at which V compression becomes viable
(PPL gap < 0.05x above K-only reference).

Design:
  - K always at k=112/4-bit (our known good setting from exps 7, 15, 17)
  - K-only reference: K compressed, V at full precision
  - V scan: k_V = {64, 80, 96, 104, 108, 112, 116, 120, 124, 128}
    * k=128 = full subspace (all dims kept, just 4-bit quantization, no projection)
  - Also test: k_V=112 at 8-bit to compare bits vs dimensions tradeoff
  - Eval ctx = 4096 (sufficient to measure compression quality)
  - Also eval at ctx=8192 for the viable k values

  Threshold: k_V is "viable" if PPL_ratio(K+V) < PPL_ratio(K_only) + 0.05
  i.e., adding V compression costs < 5% additional PPL degradation

Outputs:
  results/exp20_v_threshold.csv
  results/REPORT-20-v-threshold.md
"""

import sys
import csv
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compress import polar_quantize, subspace_polar_quantize, fit_pca
from collect import get_model_and_tokenizer, find_attention_layers

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── Data / collection helpers ─────────────────────────────────────────────────

def load_tokens(tokenizer, data_file, char_offset, n_tokens):
    with open(data_file, 'r', encoding='utf-8') as f:
        f.seek(char_offset)
        text = f.read(n_tokens * 6)
    ids = tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
    device = torch.device("cuda:0")
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)


def collect_kvs_for_basis(model, tokenizer, data_file, char_offset, n_tokens,
                           n_kv_heads, d_head):
    """Returns {(layer_idx, head_idx): {'K': np.array(T, d_head), 'V': np.array(T, d_head)}}"""
    input_ids = load_tokens(tokenizer, data_file, char_offset, n_tokens)
    if input_ids.shape[1] > n_tokens:
        input_ids = input_ids[:, :n_tokens]

    kv_store = {}
    hooks = []
    attn_layers = find_attention_layers(model)
    for layer_idx, attn in attn_layers:
        for kv_type, proj_name in [('K', 'k_proj'), ('V', 'v_proj')]:
            def make_capture(li, kvt, nh, dh):
                def hook(module, input, output):
                    x = output.detach().cpu().float()
                    b, s, _ = x.shape
                    x = x.reshape(b, s, nh, dh)[0]
                    key = (li, kvt)
                    if key not in kv_store:
                        kv_store[key] = []
                    kv_store[key].append(x.numpy())
                return hook
            h = getattr(attn, proj_name).register_forward_hook(
                make_capture(layer_idx, kv_type, n_kv_heads, d_head))
            hooks.append(h)

    with torch.no_grad():
        model(input_ids=input_ids)
    for h in hooks:
        h.remove()

    bases_raw = {}
    for (layer_idx, kv_type), arrays in kv_store.items():
        arr = np.concatenate(arrays, axis=0)   # (T, n_kv_heads, d_head)
        for head_idx in range(arr.shape[1]):
            key = (layer_idx, head_idx)
            if key not in bases_raw:
                bases_raw[key] = {}
            bases_raw[key][kv_type] = arr[:, head_idx, :]
    return bases_raw

MODEL_NAME   = "Qwen/Qwen3-14B-AWQ"
DATA_FILE    = Path("data/war_and_peace.txt")
CALIB_TOKENS = 2048
CALIB_OFFSET = 0
EVAL_OFFSET  = 5000
K_DIM_K      = 112   # K dimension — fixed throughout
BITS         = 4

# V k candidates: coarse at low end, fine-grained near k=112
V_K_CANDIDATES = [64, 80, 96, 104, 108, 112, 116, 120, 124, 128]
# Also test 8-bit at k=112 for bits-vs-dims tradeoff
V_8BIT_K       = 112
V_8BIT_BITS    = 8

# Eval contexts: short for full scan, long for viable-only
EVAL_CTX_SHORT = 4096
EVAL_CTX_LONG  = 8192
VIABLE_THRESHOLD = 0.05   # max additional rel PPL above K-only


# ── Basis fitting ─────────────────────────────────────────────────────────────

def build_bases(kvs, d_head=128):
    """Build per-(layer, head) full-rank PCA bases from collected KV data.
    Stores all d_head components — hooks select how many to use at runtime."""
    k_bases, v_bases = {}, {}
    for (layer_idx, head_idx), data in kvs.items():
        K_data = data['K']   # (T, d_head)
        V_data = data['V']
        U_k, mean_k = fit_pca(K_data, d_head)
        U_v, mean_v = fit_pca(V_data, d_head)
        k_bases[(layer_idx, head_idx)] = (U_k, mean_k)
        v_bases[(layer_idx, head_idx)] = (U_v, mean_v)
    return k_bases, v_bases


# ── Hook installation ─────────────────────────────────────────────────────────

def install_kv_hooks(model, k_bases, v_bases, k_dim_k, k_dim_v, bits_k, bits_v,
                     n_kv_heads, d_head, compress_v=True):
    """
    Install K and V compression hooks.
    - K: always compressed at k_dim_k/bits_k
    - V: compressed at k_dim_v/bits_v if compress_v=True, else full-dim quantize at bits_v
    """
    hooks = []
    attn_layers = find_attention_layers(model)

    for layer_idx, attn in attn_layers:
        def make_k_hook(li):
            def hook(module, input, output):
                dev, dty = output.device, output.dtype
                x = output.detach().cpu().float()
                b, s, _ = x.shape
                xr = x.reshape(b, s, n_kv_heads, d_head)
                for h in range(n_kv_heads):
                    xh = xr[0, :, h, :].numpy()
                    U_full, mean = k_bases[(li, h)]
                    U = U_full[:, :k_dim_k]  # slice to (d, k_dim_k)
                    xr[0, :, h, :] = torch.from_numpy(
                        subspace_polar_quantize(xh, k_dim_k, bits_k, U, mean))
                return xr.reshape(b, s, n_kv_heads * d_head).to(dty).to(dev)
            return hook

        def make_v_hook_compressed(li, k_dim_v_, bits_v_):
            def hook(module, input, output):
                dev, dty = output.device, output.dtype
                x = output.detach().cpu().float()
                b, s, _ = x.shape
                xr = x.reshape(b, s, n_kv_heads, d_head)
                for h in range(n_kv_heads):
                    xh = xr[0, :, h, :].numpy()
                    U_full, mean = v_bases[(li, h)]
                    U = U_full[:, :k_dim_v_]  # slice to (d, k_dim_v_)
                    xr[0, :, h, :] = torch.from_numpy(
                        subspace_polar_quantize(xh, k_dim_v_, bits_v_, U, mean))
                return xr.reshape(b, s, n_kv_heads * d_head).to(dty).to(dev)
            return hook

        def make_v_hook_full_dim(li_fd, bits_v_):
            """V at full dimension (k=128), just quantize."""
            def hook(module, input, output):
                dev, dty = output.device, output.dtype
                x = output.detach().cpu().float()
                b, s, _ = x.shape
                xr = x.reshape(b, s, n_kv_heads, d_head)
                for h in range(n_kv_heads):
                    xh = xr[0, :, h, :].numpy()
                    # Full-dim: use full PCA basis (k=d_head), just quantize
                    U_full, mean = v_bases[(li_fd, h)]
                    # U_full is (d, d) — pass as-is, k=d_head
                    xr[0, :, h, :] = torch.from_numpy(
                        subspace_polar_quantize(xh, d_head, bits_v_, U_full, mean))
                return xr.reshape(b, s, n_kv_heads * d_head).to(dty).to(dev)
            return hook

        hooks.append(attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))

        if compress_v:
            if k_dim_v == 128:
                hooks.append(attn.v_proj.register_forward_hook(
                    make_v_hook_full_dim(layer_idx, bits_v)))
            else:
                hooks.append(attn.v_proj.register_forward_hook(
                    make_v_hook_compressed(layer_idx, k_dim_v, bits_v)))
        # else: no V hook = V at full precision (K-only mode)

    return hooks


# ── PPL evaluation ────────────────────────────────────────────────────────────


def chunked_cross_entropy(model, input_ids, chunk_size=256):
    """Compute PPL via chunked logit projection to avoid OOM."""
    body   = model.model.model
    lm_head = model.model.lm_head
    device = next(model.parameters()).device

    input_ids = input_ids.to(device)
    with torch.no_grad():
        hidden = body(input_ids=input_ids).last_hidden_state  # [1, T, d]
    T = hidden.shape[1]
    total_loss = 0.0
    n = 0
    for start in range(0, T - 1, chunk_size):
        end = min(start + chunk_size, T - 1)
        h_chunk = hidden[0, start:end, :]         # [chunk, d]
        logits  = lm_head(h_chunk)                 # [chunk, vocab]
        targets = input_ids[0, start + 1:end + 1]  # [chunk]
        loss = torch.nn.functional.cross_entropy(logits, targets)
        total_loss += loss.item() * (end - start)
        n += (end - start)
    return float(np.exp(total_loss / n))


def eval_ppl(model, tokenizer, data_file, char_offset, n_tokens, device):
    input_ids = load_tokens(tokenizer, data_file, char_offset, n_tokens)
    return chunked_cross_entropy(model, input_ids)


# ── Compression ratio helper ──────────────────────────────────────────────────

def compression_ratio(k_dim_k, bits_k, k_dim_v, bits_v, d_head=128, bits_full=16):
    """Theoretical KV cache compression ratio vs fp16."""
    orig = 2 * d_head * bits_full          # K + V at fp16
    comp = k_dim_k * bits_k + k_dim_v * bits_v
    return orig / comp


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Exp 20: V-specific k Threshold Scan ===")

    device = torch.device("cuda:0")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    model.eval()

    # Get model dims
    attn_layers = find_attention_layers(model)
    first_attn  = attn_layers[0][1]
    n_kv_heads  = first_attn.k_proj.out_features // (
        first_attn.k_proj.in_features // (
            model.model.model.config.num_key_value_heads or 8))
    # Safer: read from config directly
    cfg = model.model.model.config
    n_kv_heads = cfg.num_key_value_heads
    d_head     = cfg.hidden_size // cfg.num_attention_heads
    print(f"  n_kv_heads={n_kv_heads}, d_head={d_head}")

    # Calibrate
    print(f"\nCollecting KV basis on {CALIB_TOKENS} tokens...")
    kvs = collect_kvs_for_basis(
        model, tokenizer, DATA_FILE, CALIB_OFFSET, CALIB_TOKENS,
        n_kv_heads=n_kv_heads, d_head=d_head
    )
    k_bases, v_bases = build_bases(kvs, d_head=d_head)
    print(f"  Built bases for {len(k_bases)} (layer, head) pairs")

    # Baseline
    print(f"\nComputing baseline PPL (ctx={EVAL_CTX_SHORT})...")
    baseline_ppl = eval_ppl(model, tokenizer, DATA_FILE, EVAL_OFFSET, EVAL_CTX_SHORT, device)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    # K-only reference
    print(f"\nK-only reference (K=k{K_DIM_K}/{BITS}bit, V full precision)...")
    k_hooks = install_kv_hooks(model, k_bases, v_bases,
                               K_DIM_K, 128, BITS, BITS,
                               n_kv_heads, d_head, compress_v=False)
    k_only_ppl = eval_ppl(model, tokenizer, DATA_FILE, EVAL_OFFSET, EVAL_CTX_SHORT, device)
    for h in k_hooks: h.remove()
    k_only_rel = k_only_ppl / baseline_ppl
    print(f"  K-only PPL={k_only_ppl:.4f} rel={k_only_rel:.4f}")

    viable_threshold = k_only_rel + VIABLE_THRESHOLD
    print(f"  Viability threshold: rel PPL < {viable_threshold:.4f}")

    rows = []
    viable_ks = []

    print(f"\n{'─'*60}")
    print(f"V k scan at ctx={EVAL_CTX_SHORT} (K fixed at k{K_DIM_K}/{BITS}bit)")
    print(f"{'─'*60}")

    for k_v in V_K_CANDIDATES:
        hooks = install_kv_hooks(model, k_bases, v_bases,
                                 K_DIM_K, k_v, BITS, BITS,
                                 n_kv_heads, d_head, compress_v=True)
        ppl = eval_ppl(model, tokenizer, DATA_FILE, EVAL_OFFSET, EVAL_CTX_SHORT, device)
        for h in hooks: h.remove()

        rel       = ppl / baseline_ppl
        gap       = rel - k_only_rel
        cr        = compression_ratio(K_DIM_K, BITS, k_v, BITS, d_head)
        viable    = rel < viable_threshold
        if viable:
            viable_ks.append(k_v)

        print(f"  k_V={k_v:3d}  PPL={ppl:.4f}  rel={rel:.4f}  "
              f"gap={gap:+.4f}  CR={cr:.2f}x  {'✓ VIABLE' if viable else ''}")

        rows.append({
            'experiment': 'v_k_scan',
            'k_K': K_DIM_K,
            'k_V': k_v,
            'bits_K': BITS,
            'bits_V': BITS,
            'eval_ctx': EVAL_CTX_SHORT,
            'ppl': ppl,
            'rel_ppl': rel,
            'gap_vs_k_only': gap,
            'compression_ratio': cr,
            'viable': viable,
            'baseline_ppl': baseline_ppl,
            'k_only_ppl': k_only_ppl,
        })

    # 8-bit V test for bits-vs-dims tradeoff
    print(f"\n{'─'*60}")
    print(f"Bits vs dims: k_V={V_8BIT_K}/8-bit vs best 4-bit k_V")
    print(f"{'─'*60}")
    hooks = install_kv_hooks(model, k_bases, v_bases,
                             K_DIM_K, V_8BIT_K, BITS, V_8BIT_BITS,
                             n_kv_heads, d_head, compress_v=True)
    ppl_8bit = eval_ppl(model, tokenizer, DATA_FILE, EVAL_OFFSET, EVAL_CTX_SHORT, device)
    for h in hooks: h.remove()
    rel_8bit = ppl_8bit / baseline_ppl
    gap_8bit = rel_8bit - k_only_rel
    cr_8bit  = compression_ratio(K_DIM_K, BITS, V_8BIT_K, V_8BIT_BITS, d_head)
    print(f"  k_V={V_8BIT_K}/8bit  PPL={ppl_8bit:.4f}  rel={rel_8bit:.4f}  "
          f"gap={gap_8bit:+.4f}  CR={cr_8bit:.2f}x")
    rows.append({
        'experiment': 'bits_tradeoff_8bit',
        'k_K': K_DIM_K,
        'k_V': V_8BIT_K,
        'bits_K': BITS,
        'bits_V': V_8BIT_BITS,
        'eval_ctx': EVAL_CTX_SHORT,
        'ppl': ppl_8bit,
        'rel_ppl': rel_8bit,
        'gap_vs_k_only': gap_8bit,
        'compression_ratio': cr_8bit,
        'viable': rel_8bit < viable_threshold,
        'baseline_ppl': baseline_ppl,
        'k_only_ppl': k_only_ppl,
    })

    # Long context eval for viable k values
    if viable_ks:
        print(f"\n{'─'*60}")
        print(f"Long-context eval (ctx={EVAL_CTX_LONG}) for viable k_V: {viable_ks}")
        print(f"{'─'*60}")

        # Baseline at long ctx
        baseline_ppl_long = eval_ppl(model, tokenizer, DATA_FILE, EVAL_OFFSET,
                                     EVAL_CTX_LONG, device)
        hooks = install_kv_hooks(model, k_bases, v_bases,
                                 K_DIM_K, 128, BITS, BITS,
                                 n_kv_heads, d_head, compress_v=False)
        k_only_ppl_long = eval_ppl(model, tokenizer, DATA_FILE, EVAL_OFFSET,
                                   EVAL_CTX_LONG, device)
        for h in hooks: h.remove()
        k_only_rel_long = k_only_ppl_long / baseline_ppl_long
        viable_thresh_long = k_only_rel_long + VIABLE_THRESHOLD

        print(f"  Baseline PPL (long): {baseline_ppl_long:.4f}")
        print(f"  K-only PPL (long): {k_only_ppl_long:.4f}  rel={k_only_rel_long:.4f}")

        for k_v in viable_ks:
            hooks = install_kv_hooks(model, k_bases, v_bases,
                                     K_DIM_K, k_v, BITS, BITS,
                                     n_kv_heads, d_head, compress_v=True)
            ppl_l = eval_ppl(model, tokenizer, DATA_FILE, EVAL_OFFSET,
                             EVAL_CTX_LONG, device)
            for h in hooks: h.remove()
            rel_l   = ppl_l / baseline_ppl_long
            gap_l   = rel_l - k_only_rel_long
            viable_l = rel_l < viable_thresh_long
            cr_l    = compression_ratio(K_DIM_K, BITS, k_v, BITS, d_head)
            print(f"  k_V={k_v:3d}  PPL={ppl_l:.4f}  rel={rel_l:.4f}  "
                  f"gap={gap_l:+.4f}  CR={cr_l:.2f}x  {'✓ VIABLE' if viable_l else '✗ fails at long ctx'}")
            rows.append({
                'experiment': 'v_k_scan_long_ctx',
                'k_K': K_DIM_K,
                'k_V': k_v,
                'bits_K': BITS,
                'bits_V': BITS,
                'eval_ctx': EVAL_CTX_LONG,
                'ppl': ppl_l,
                'rel_ppl': rel_l,
                'gap_vs_k_only': gap_l,
                'compression_ratio': cr_l,
                'viable': viable_l,
                'baseline_ppl': baseline_ppl_long,
                'k_only_ppl': k_only_ppl_long,
            })

    # ── Write CSV ──────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "exp20_v_threshold.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults: {csv_path}")

    # ── Write report ──────────────────────────────────────────────────────────
    report_path = RESULTS_DIR / "REPORT-20-v-threshold.md"
    short_rows = [r for r in rows if r['experiment'] == 'v_k_scan']
    min_viable = min((r['k_V'] for r in short_rows if r['viable']), default=None)

    with open(report_path, 'w') as f:
        f.write(f"# Experiment 20: V-specific k Threshold Scan\n\n")
        f.write(f"**Model:** {MODEL_NAME}  \n")
        f.write(f"**K fixed at:** k={K_DIM_K}/{BITS}bit  \n")
        f.write(f"**Viability threshold:** gap_vs_K_only < {VIABLE_THRESHOLD:.2f}x PPL  \n")
        f.write(f"**Baseline PPL (ctx={EVAL_CTX_SHORT}):** {baseline_ppl:.4f}  \n")
        f.write(f"**K-only ref PPL:** {k_only_ppl:.4f} (rel={k_only_rel:.4f})\n\n")

        f.write(f"## V k Scan Results (ctx={EVAL_CTX_SHORT})\n\n")
        f.write(f"| k_V | PPL | Rel PPL | Gap vs K-only | CR | Viable |\n")
        f.write(f"|-----|-----|---------|---------------|-----|--------|\n")
        for r in short_rows:
            f.write(f"| {r['k_V']} | {r['ppl']:.4f} | {r['rel_ppl']:.4f} | "
                    f"{r['gap_vs_k_only']:+.4f} | {r['compression_ratio']:.2f}x | "
                    f"{'✓' if r['viable'] else '✗'} |\n")

        # 8-bit row
        r8 = next((r for r in rows if r['experiment'] == 'bits_tradeoff_8bit'), None)
        if r8:
            f.write(f"\n## Bits vs Dimensions Tradeoff\n\n")
            f.write(f"| Config | PPL | Rel PPL | Gap | CR |\n")
            f.write(f"|--------|-----|---------|-----|----|\n")
            f.write(f"| k_V={r8['k_V']}/4-bit | "
                    f"{next(r['ppl'] for r in short_rows if r['k_V']==r8['k_V']):.4f} | "
                    f"{next(r['rel_ppl'] for r in short_rows if r['k_V']==r8['k_V']):.4f} | "
                    f"{next(r['gap_vs_k_only'] for r in short_rows if r['k_V']==r8['k_V']):+.4f} | "
                    f"{compression_ratio(K_DIM_K, BITS, r8['k_V'], BITS, d_head):.2f}x |\n")
            f.write(f"| k_V={r8['k_V']}/8-bit | {r8['ppl']:.4f} | {r8['rel_ppl']:.4f} | "
                    f"{r8['gap_vs_k_only']:+.4f} | {r8['compression_ratio']:.2f}x |\n")

        if min_viable is not None:
            long_rows = [r for r in rows if r['experiment'] == 'v_k_scan_long_ctx']
            f.write(f"\n## Summary\n\n")
            f.write(f"- **Minimum viable k_V (ctx={EVAL_CTX_SHORT}):** {min_viable} "
                    f"({min_viable/d_head*100:.0f}% of d_head)\n")
            if long_rows:
                min_viable_long = min((r['k_V'] for r in long_rows if r['viable']), default=None)
                f.write(f"- **Minimum viable k_V (ctx={EVAL_CTX_LONG}):** "
                        f"{min_viable_long if min_viable_long else 'none found'}\n")
            best_cr = max(r['compression_ratio'] for r in short_rows if r['viable']) if min_viable else 0
            f.write(f"- **Best CR at viable threshold:** {best_cr:.2f}x\n")
            f.write(f"- **Combined K+V CR at k_K={K_DIM_K}/k_V={min_viable}/4bit:** "
                    f"{compression_ratio(K_DIM_K, BITS, min_viable, BITS, d_head):.2f}x\n")
        else:
            f.write(f"\n## Summary\n\nNo viable k_V found below d_head={d_head}. "
                    f"V compression at 4-bit is not viable for this model at any "
                    f"subspace dimension. Consider 8-bit quantization for V.\n")

        f.write(f"\n## Interpretation\n\n")
        f.write(f"Exp 13 found V basis overlap = 0.702 vs K = 0.825. This means ~30% of V variance\n")
        f.write(f"lives in dimensions beyond k=90. The threshold scan above identifies the exact\n")
        f.write(f"k_V at which V compression becomes practical for deployment.\n")

    print(f"Report:  {report_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
