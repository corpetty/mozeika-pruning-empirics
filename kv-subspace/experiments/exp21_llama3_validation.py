"""
Experiment 21: Llama-3.1 Architecture Validation

Tests whether the subspace compression findings from Qwen3/Mistral/Phi3 hold on
Llama-3.1-8B-Instruct-AWQ (llama architecture, 32 layers, 8 KV heads, d_head=128).

Llama-3.1 uses grouped-query attention (GQA) with 32 Q heads / 8 KV heads,
RoPE positional encoding, no QK-norm (unlike Qwen3). This makes it a clean
comparison point for the QK-norm hypothesis about V compression.

Sub-experiments:
  A. K+V sweep: k/d_head = {0.50, 0.75, 0.875, 0.9375, 1.0} at 4-bit
     — establishes threshold for Llama-3.1, compare vs Qwen3-14B / Mistral-7B
  B. K-only vs V-only at k=112 (best practical k):
     — isolates K vs V contribution to compression cost
  C. V threshold scan: k_V in {64, 96, 112, 120, 124, 128} with K fixed at k=112
     — tests QK-norm hypothesis: Llama has no QK-norm, should V compress better?

Model: hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
  n_layers=32, n_kv_heads=8, d_head=128 (4096 hidden / 32 heads)
"""

import sys
import os
import gc
import csv
import json
import numpy as np
import torch
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compress import polar_quantize, subspace_polar_quantize, fit_pca, quantize_uniform
from collect import find_attention_layers, get_sample_text

MODEL_ID   = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
N_LAYERS   = 32
N_KV_HEADS = 8
D_HEAD     = 128

# Same eval passages as all prior experiments
EVAL_PASSAGES = [
    (
        "The mitochondria are membrane-bound organelles found in the cytoplasm of "
        "eukaryotic cells. They generate most of the cell's supply of adenosine "
        "triphosphate, used as a source of chemical energy. Mitochondria have their "
        "own DNA, known as mitochondrial DNA, which is separate from the nuclear DNA "
        "found in the cell nucleus. This organelle has its own ribosomes and can "
        "synthesize some of its own proteins. The number of mitochondria in a cell "
        "varies widely by organism and tissue type. Many cells have only a single "
        "mitochondrion, whereas others can contain several thousand mitochondria. "
        "The organelle is composed of compartments that carry out specialized "
        "functions. These compartments or regions include the outer membrane, the "
        "intermembrane space, the inner membrane, the cristae, and the matrix. "
        "Although most of a cell's DNA is contained in the cell nucleus, the "
        "mitochondrion has its own genome that is substantially different from the "
        "nuclear genome."
    ),
    (
        "In computer science, a binary search tree is a rooted binary tree data "
        "structure with the key of each internal node being greater than all the "
        "keys in the respective node's left subtree and less than the ones in its "
        "right subtree. The time complexity of operations on the binary search tree "
        "is linear with respect to the height of the tree. Binary search trees allow "
        "binary search for fast lookup, addition, and removal of data items. Since "
        "the nodes in a BST are laid out so that each comparison skips about half of "
        "the remaining tree, the lookup performance is proportional to that of binary "
        "logarithm. BSTs were devised in the 1960s for the problem of efficient "
        "storage of labeled data and are attributed to Conway Berners-Lee and David "
        "Wheeler. The performance of a binary search tree is dependent on the order "
        "of insertion of the nodes into the tree since arbitrary insertions may lead "
        "to degeneracy; several variations of the basic data structure exist that "
        "impose limits on growth and therefore provide guaranteed upper bounds on "
        "tree performance."
    ),
    (
        "The French Revolution was a period of radical political and societal change "
        "in France that began with the Estates General of 1789 and ended with the "
        "formation of the French Consulate in November 1799. Many of its ideas are "
        "considered fundamental principles of liberal democracy, while phrases like "
        "liberté, égalité, fraternité reappeared in other revolts, such as the 1917 "
        "Russian Revolution, and inspired campaigns for the abolition of slavery and "
        "universal suffrage. The values and institutions of the Revolution dominate "
        "French politics to this day. Its causes are generally agreed to be a "
        "combination of social, political, and economic factors, which the Ancien "
        "Régime proved unable to manage. In May 1789, widespread social distress led "
        "to the convocation of the Estates General, which was converted into a "
        "National Assembly in June. The storming of the Bastille on 14 July led to a "
        "series of radical measures by the Assembly, abolishing feudalism and "
        "proclaiming the Declaration of the Rights of Man and of the Citizen."
    ),
]


# ── Model loading ────────────────────────────────────────────────────────────

def load_model():
    from transformers import AutoTokenizer
    from awq import AutoAWQForCausalLM
    print(f"Loading {MODEL_ID} ...")
    model = AutoAWQForCausalLM.from_quantized(
        MODEL_ID,
        fuse_layers=False,
        trust_remote_code=False,
        safetensors=True,
        max_memory={0: "20GiB", "cpu": "40GiB"},
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    # Llama AWQ: model body is model.model (not model.model.model like Qwen3)
    print(f"  Model type: {type(model).__name__}")
    print(f"  n_kv_heads={N_KV_HEADS}, d_head={D_HEAD}, n_layers={N_LAYERS}")
    return model, tokenizer


# ── KV basis collection ───────────────────────────────────────────────────────

def collect_kvs_for_basis(model, tokenizer, text, n_tokens, device):
    """
    Returns {(layer_idx, head_idx): {'K': np.array(T, d_head), 'V': ...}}
    Uses forward hooks — same pattern as all prior experiments.
    """
    kvs = {}
    hooks = []
    attn_layers = find_attention_layers(model)

    for layer_idx, attn in attn_layers:
        def make_hook(li):
            def hook(module, input, output):
                x = output.detach().cpu().float()
                b, s, _ = x.shape
                # Llama v_proj output: (b, s, n_kv_heads * d_head)
                xr = x.reshape(b, s, N_KV_HEADS, D_HEAD)
                for h in range(N_KV_HEADS):
                    key = (li, h)
                    if key not in kvs:
                        kvs[key] = {'K': [], 'V': []}
                    kvs[key]['V'].append(xr[0, :, h, :].numpy())
            return hook

        def make_k_hook(li):
            def hook(module, input, output):
                x = output.detach().cpu().float()
                b, s, _ = x.shape
                xr = x.reshape(b, s, N_KV_HEADS, D_HEAD)
                for h in range(N_KV_HEADS):
                    key = (li, h)
                    if key not in kvs:
                        kvs[key] = {'K': [], 'V': []}
                    kvs[key]['K'].append(xr[0, :, h, :].numpy())
            return hook

        hooks.append(attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))
        hooks.append(attn.v_proj.register_forward_hook(make_hook(layer_idx)))

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=n_tokens)
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    # Concatenate across chunks
    result = {}
    for key, d in kvs.items():
        result[key] = {
            'K': np.concatenate(d['K'], axis=0),
            'V': np.concatenate(d['V'], axis=0),
        }
    return result


def build_bases(kvs):
    """Fit full-rank PCA bases. Returns {(li, hi): (U_full_d_head, mean)}."""
    bases_k, bases_v = {}, {}
    for (li, hi), d in kvs.items():
        Uk, mk = fit_pca(d['K'], D_HEAD)
        Uv, mv = fit_pca(d['V'], D_HEAD)
        bases_k[(li, hi)] = (Uk, mk)
        bases_v[(li, hi)] = (Uv, mv)
    return bases_k, bases_v


# ── Chunked cross-entropy (avoid 9GB logit tensor at large ctx) ──────────────

def chunked_cross_entropy(model, input_ids, chunk_size=512):
    """Run model body once, project hidden states in chunks."""
    # Llama AWQ: model body is model.model (not model.model.model)
    body   = model.model
    lm_head = model.lm_head
    device  = input_ids.device

    with torch.no_grad():
        out = body(input_ids, output_hidden_states=False)
        hidden = out.last_hidden_state  # (1, T, d_model)

    T = hidden.shape[1]
    total_loss = 0.0
    n_tokens   = 0
    labels     = input_ids[0]  # (T,)

    for start in range(0, T - 1, chunk_size):
        end = min(start + chunk_size, T - 1)
        h_chunk  = hidden[0, start:end, :]          # (chunk, d_model)
        logits   = lm_head(h_chunk)                  # (chunk, vocab)
        tgt      = labels[start + 1: end + 1]        # (chunk,)
        loss     = torch.nn.functional.cross_entropy(logits, tgt, reduction='sum')
        total_loss += loss.item()
        n_tokens   += end - start

    return total_loss / n_tokens


def eval_ppl(model, tokenizer, passages, device, ctx=512):
    total_loss = 0.0
    n = 0
    for text in passages:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=ctx)
        ids = enc["input_ids"].to(device)
        if ids.shape[1] < 2:
            continue
        loss = chunked_cross_entropy(model, ids)
        total_loss += loss
        n += 1
    import math
    return math.exp(total_loss / n)


# ── Hook installation ─────────────────────────────────────────────────────────

def install_hooks(model, bases_k, bases_v, k_dim_k, bits_k,
                  compress_v=True, k_dim_v=None, bits_v=4):
    """
    Install forward hooks compressing K (always) and optionally V.
    k_dim_k, k_dim_v: subspace dimension. If == D_HEAD, uses full-dim polar_quantize.
    """
    if k_dim_v is None:
        k_dim_v = k_dim_k

    hooks = []
    attn_layers = find_attention_layers(model)

    for layer_idx, attn in attn_layers:

        def make_k_hook(li):
            def hook(module, input, output):
                dev, dty = output.device, output.dtype
                x = output.detach().cpu().float()
                b, s, _ = x.shape
                xr = x.reshape(b, s, N_KV_HEADS, D_HEAD)
                for h in range(N_KV_HEADS):
                    xh = xr[0, :, h, :].numpy()
                    U_full, mean = bases_k[(li, h)]
                    if k_dim_k == D_HEAD:
                        xr[0, :, h, :] = torch.from_numpy(
                            subspace_polar_quantize(xh, D_HEAD, bits_k, U_full, mean))
                    else:
                        U = U_full[:, :k_dim_k]
                        xr[0, :, h, :] = torch.from_numpy(
                            subspace_polar_quantize(xh, k_dim_k, bits_k, U, mean))
                return xr.reshape(b, s, N_KV_HEADS * D_HEAD).to(dty).to(dev)
            return hook

        def make_v_hook(li, k_v, bits_v_):
            def hook(module, input, output):
                dev, dty = output.device, output.dtype
                x = output.detach().cpu().float()
                b, s, _ = x.shape
                xr = x.reshape(b, s, N_KV_HEADS, D_HEAD)
                for h in range(N_KV_HEADS):
                    xh = xr[0, :, h, :].numpy()
                    U_full, mean = bases_v[(li, h)]
                    if k_v == D_HEAD:
                        xr[0, :, h, :] = torch.from_numpy(
                            subspace_polar_quantize(xh, D_HEAD, bits_v_, U_full, mean))
                    else:
                        U = U_full[:, :k_v]
                        xr[0, :, h, :] = torch.from_numpy(
                            subspace_polar_quantize(xh, k_v, bits_v_, U, mean))
                return xr.reshape(b, s, N_KV_HEADS * D_HEAD).to(dty).to(dev)
            return hook

        hooks.append(attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))
        if compress_v:
            hooks.append(attn.v_proj.register_forward_hook(
                make_v_hook(layer_idx, k_dim_v, bits_v)))

    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ── Main ──────────────────────────────────────────────────────────────────────

def compression_ratio(k, bits, d=D_HEAD, fp_bits=16):
    return (d * fp_bits) / (k * bits)


def main():
    print("=== Exp 21: Llama-3.1 Architecture Validation ===")
    device = "cuda:0"

    model, tokenizer = load_model()
    model = model.to(device) if hasattr(model, 'to') else model

    # Calibration text: 2048 tokens of mixed prose
    calib_text = " ".join([p * 3 for p in EVAL_PASSAGES])
    print("\nCollecting KV bases on 2048 calibration tokens...")
    kvs = collect_kvs_for_basis(model, tokenizer, calib_text, 2048, device)
    bases_k, bases_v = build_bases(kvs)
    del kvs
    gc.collect()
    print(f"  Built bases for {len(bases_k)} (layer, head) pairs")

    # Baseline PPL
    print("\nComputing baseline PPL (ctx=512)...")
    baseline_ppl = eval_ppl(model, tokenizer, EVAL_PASSAGES, device, ctx=512)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    results = []

    # ── Sub-exp A: K+V sweep ─────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("Sub-exp A: K+V sweep (same k for K and V)")
    print("─"*60)

    k_fracs = [0.50, 0.75, 0.875, 0.9375, 1.0]
    k_vals  = [int(D_HEAD * f) for f in k_fracs]  # 64, 96, 112, 120, 128

    for k in k_vals:
        hooks = install_hooks(model, bases_k, bases_v, k, 4,
                              compress_v=True, k_dim_v=k, bits_v=4)
        ppl = eval_ppl(model, tokenizer, EVAL_PASSAGES, device, ctx=512)
        remove_hooks(hooks)
        rel = ppl / baseline_ppl
        cr  = compression_ratio(k, 4)
        frac = k / D_HEAD
        print(f"  k={k:3d} (k/d={frac:.4f})  PPL={ppl:.4f}  rel={rel:.4f}  CR={cr:.2f}x")
        results.append({
            'subexp': 'A_kv_sweep',
            'k_K': k, 'bits_K': 4, 'k_V': k, 'bits_V': 4,
            'compress_v': True, 'ctx': 512,
            'baseline_ppl': baseline_ppl, 'ppl': ppl,
            'rel_ppl': rel, 'compression_ratio': cr,
        })
        gc.collect()

    # ── Sub-exp B: K-only vs V-only at k=112 ────────────────────────────────
    print("\n" + "─"*60)
    print("Sub-exp B: K-only vs V-only at k=112/4-bit")
    print("─"*60)

    # K-only (no V compression)
    hooks = install_hooks(model, bases_k, bases_v, 112, 4, compress_v=False)
    ppl_k_only = eval_ppl(model, tokenizer, EVAL_PASSAGES, device, ctx=512)
    remove_hooks(hooks)
    rel_k = ppl_k_only / baseline_ppl
    print(f"  K-only k=112/4bit  PPL={ppl_k_only:.4f}  rel={rel_k:.4f}")
    results.append({
        'subexp': 'B_k_only', 'k_K': 112, 'bits_K': 4, 'k_V': None, 'bits_V': None,
        'compress_v': False, 'ctx': 512,
        'baseline_ppl': baseline_ppl, 'ppl': ppl_k_only,
        'rel_ppl': rel_k, 'compression_ratio': compression_ratio(112, 4),
    })
    gc.collect()

    # V-only (K at full precision)
    # Abuse install_hooks: set k_K=128 (no subspace reduction) but compress_v=True k_V=112
    hooks = install_hooks(model, bases_k, bases_v, 128, 16, compress_v=True, k_dim_v=112, bits_v=4)
    ppl_v_only = eval_ppl(model, tokenizer, EVAL_PASSAGES, device, ctx=512)
    remove_hooks(hooks)
    rel_v = ppl_v_only / baseline_ppl
    print(f"  V-only k=112/4bit  PPL={ppl_v_only:.4f}  rel={rel_v:.4f}")
    results.append({
        'subexp': 'B_v_only', 'k_K': None, 'bits_K': None, 'k_V': 112, 'bits_V': 4,
        'compress_v': True, 'ctx': 512,
        'baseline_ppl': baseline_ppl, 'ppl': ppl_v_only,
        'rel_ppl': rel_v, 'compression_ratio': compression_ratio(112, 4),
    })
    gc.collect()

    # ── Sub-exp C: V threshold scan with K fixed at k=112 ────────────────────
    print("\n" + "─"*60)
    print("Sub-exp C: V threshold scan (K fixed at k=112/4bit)")
    print("Sub-test of QK-norm hypothesis: Llama3 has no QK-norm, V should compress better")
    print("─"*60)

    viability_threshold = rel_k + 0.05
    print(f"  K-only reference rel={rel_k:.4f}, viability threshold rel < {viability_threshold:.4f}")

    v_k_vals = [64, 96, 112, 120, 124, 128]
    for k_v in v_k_vals:
        hooks = install_hooks(model, bases_k, bases_v, 112, 4,
                              compress_v=True, k_dim_v=k_v, bits_v=4)
        ppl = eval_ppl(model, tokenizer, EVAL_PASSAGES, device, ctx=512)
        remove_hooks(hooks)
        rel = ppl / baseline_ppl
        gap = rel - rel_k
        cr  = compression_ratio(k_v, 4)
        viable = "✓ VIABLE" if gap < 0.05 else "✗"
        print(f"  k_V={k_v:3d}  PPL={ppl:.4f}  rel={rel:.4f}  gap={gap:+.4f}  CR={cr:.2f}x  {viable}")
        results.append({
            'subexp': 'C_v_threshold',
            'k_K': 112, 'bits_K': 4, 'k_V': k_v, 'bits_V': 4,
            'compress_v': True, 'ctx': 512,
            'baseline_ppl': baseline_ppl, 'ppl': ppl,
            'rel_ppl': rel, 'compression_ratio': cr,
            'gap_vs_k_only': gap, 'viable': gap < 0.05,
        })
        gc.collect()

    # ── Save results ──────────────────────────────────────────────────────────
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "exp21_llama3_validation.csv"
    fieldnames = ['subexp', 'k_K', 'bits_K', 'k_V', 'bits_V', 'compress_v', 'ctx',
                  'baseline_ppl', 'ppl', 'rel_ppl', 'compression_ratio',
                  'gap_vs_k_only', 'viable']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    print(f"\nResults: {csv_path}")

    # ── Write report ──────────────────────────────────────────────────────────
    # Find min viable k in sub-exp A
    a_rows = [r for r in results if r['subexp'] == 'A_kv_sweep']
    viable_a = [r for r in a_rows if r['rel_ppl'] < 1.20]
    min_viable_k = min(r['k_K'] for r in viable_a) if viable_a else "none"

    # Sub-exp C: any k_V viable?
    c_rows = [r for r in results if r['subexp'] == 'C_v_threshold']
    viable_c = [r for r in c_rows if r.get('viable', False)]
    min_viable_v = min(r['k_V'] for r in viable_c) if viable_c else "none (V not viable)"

    report_path = results_dir / "REPORT-21-llama3-validation.md"
    with open(report_path, 'w') as f:
        f.write(f"""# Experiment 21: Llama-3.1 Architecture Validation

**Model:** hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4  
**Architecture:** llama (32 layers, 8 KV heads, d_head=128, no QK-norm)  
**Baseline PPL (ctx=512):** {baseline_ppl:.4f}

## Motivation

All prior experiments used Qwen3-14B-AWQ as the primary model. Qwen3 applies
QK-norm (RMSNorm on k_proj and q_proj outputs) which may force K into a
low-dimensional manifold while leaving V variance undistorted. Llama-3.1 uses
standard GQA without QK-norm — making it the critical test case for the
cross-architecture V compression hypothesis.

## Sub-exp A: K+V Sweep (k/d_head fractions)

| k | k/d_head | PPL | Rel PPL | CR | Within 20% |
|---|----------|-----|---------|-----|------------|
""")
        for r in a_rows:
            within = "✓" if r['rel_ppl'] < 1.20 else "✗"
            f.write(f"| {r['k_K']} | {r['k_K']/D_HEAD:.4f} | {r['ppl']:.4f} | "
                    f"{r['rel_ppl']:.4f} | {r['compression_ratio']:.2f}x | {within} |\n")
        f.write(f"\n**Minimum viable k for Llama-3.1-8B (< 20% PPL): {min_viable_k}**\n\n")

        b_k = next((r for r in results if r['subexp'] == 'B_k_only'), None)
        b_v = next((r for r in results if r['subexp'] == 'B_v_only'), None)
        f.write(f"""## Sub-exp B: K-only vs V-only at k=112/4-bit

| Config | PPL | Rel PPL | CR |
|--------|-----|---------|-----|
""")
        if b_k:
            f.write(f"| K-only (V full precision) | {b_k['ppl']:.4f} | {b_k['rel_ppl']:.4f} | "
                    f"{b_k['compression_ratio']:.2f}x |\n")
        if b_v:
            f.write(f"| V-only (K full precision) | {b_v['ppl']:.4f} | {b_v['rel_ppl']:.4f} | "
                    f"{b_v['compression_ratio']:.2f}x |\n")
        f.write("\n")

        f.write(f"""## Sub-exp C: V Threshold Scan (QK-norm hypothesis test)

K fixed at k=112/4-bit. Viability: gap vs K-only < 0.05x rel PPL.

| k_V | PPL | Rel PPL | Gap vs K-only | CR | Viable |
|-----|-----|---------|---------------|-----|--------|
""")
        for r in c_rows:
            viable = "✓" if r.get('viable', False) else "✗"
            f.write(f"| {r['k_V']} | {r['ppl']:.4f} | {r['rel_ppl']:.4f} | "
                    f"{r.get('gap_vs_k_only', 0):+.4f} | {r['compression_ratio']:.2f}x | {viable} |\n")

        f.write(f"""
**Minimum viable k_V for Llama-3.1-8B:** {min_viable_v}

## Cross-Architecture Comparison

| Model | Arch | Params | Min k for <20% PPL | Rel PPL at k=112 |
|-------|------|--------|-------------------|------------------|
| Qwen3-1.7B | Qwen3 (QK-norm) | 1.7B | 128 | 1.32x |
| Mistral-7B-v0.3 | Mistral (no QK-norm) | 7B | 64 | 1.07x |
| Qwen3-14B-AWQ | Qwen3 (QK-norm) | 14B | 112 | 1.14x |
| Phi-4-AWQ | Phi3 (no QK-norm) | 14B | 64 | 1.10x |
| Llama-3.1-8B | Llama (no QK-norm) | 8B | {min_viable_k} | (see above) |

## QK-norm Hypothesis Assessment

Qwen3 applies QK-norm (RMSNorm after k_proj/q_proj) but not V-norm. Llama-3.1,
Mistral-7B, and Phi-3 do not use QK-norm. The hypothesis: QK-norm forces K into
a low-dimensional manifold while V retains full variance, explaining why V
compression fails for Qwen3 but may succeed for other architectures.

If Llama-3.1 V compression is viable at k<128, this supports the QK-norm hypothesis.
If not, V compression is universally hard, and the mechanism is something else
(possibly GQA itself, since all three non-Qwen models also use GQA).
""")

    print(f"Report:  {report_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
