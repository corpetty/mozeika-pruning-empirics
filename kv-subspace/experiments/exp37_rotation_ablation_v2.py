"""
Task F1 (v2): Rotation ablation — corrected design.

PROBLEM WITH exp36
------------------
exp36's `compress_rotation_only` at k<d was broken: it embedded a k×k rotation
in the top-left of a d×d identity, then quantized only the top-k dims but kept
ALL d dims. At k=112 this stored 128 dims — no truncation — making it incomparable
to pca_only/subrotq which genuinely truncate to k dims.

CORRECTED DESIGN
----------------
We want to isolate two distinct contributions:
  1. Subspace selection (PCA truncation to k dims)
  2. Rotation before quantization (variance equalization in k-dim space)

Conditions — all at 4-bit, K-only:

  k=128 (no truncation, k=d=128):
    A. plain_4bit:      uniform quant of raw K, no PCA, no rotation
    B. raw_rotation:    apply full d×d random rotation to raw K, quantize all d dims, rotate back
    C. pca_only_128:    PCA to k=128, quantize, unproject  (= coordinate change only)
    D. subrotq_128:     PCA to k=128 + k×k rotation + quantize  (full pipeline)

  k=112 (truncation regime, k<d):
    E. pca_only_112:    PCA to k=112, quantize k dims, unproject  (truncation, no rotation)
    F. subrotq_112:     PCA to k=112 + 112×112 rotation + quantize  (truncation + rotation)

  NOTE: No "rotation_only at k=112" — there is no well-defined way to apply rotation
  with truncation without PCA; the PCA basis is what defines which dims to keep.

WHAT THIS TELLS US
------------------
  A vs B:  Does rotation help when quantizing raw K (no PCA)?
  C vs D:  Does rotation help within PCA subspace at full rank?
  E vs F:  Does rotation help within PCA subspace when truncating?
  A vs D:  Full SubRotQ pipeline vs no compression (the headline)
  A vs E:  Truncation cost without rotation
  A vs F:  Full SubRotQ at k=112 (headline for smaller k)

Output:
  results/exp37_rotation_ablation_v2.csv
  results/REPORT-37-rotation-ablation-v2.md
"""

import sys
import csv
import os
import time
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from collect import get_model_and_tokenizer
from compress import fit_pca, random_rotation_matrix

MODEL_NAME   = "Qwen/Qwen3-14B-AWQ"
RESULTS_DIR  = Path("results")
CALIB_TOKENS = 2048
EVAL_TOKENS  = 2048
N_KV_HEADS   = 8
D_HEAD       = 128
N_LAYERS     = 40
N_BITS       = 4
SEED         = 0


def get_wikitext2_tokens(tokenizer, split, n_tokens, device):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    text = "\n".join(line for line in text.split("\n") if line.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if ids.shape[1] < n_tokens + 1:
        raise ValueError(f"Only {ids.shape[1]} tokens available")
    return ids[:, :n_tokens].to(device)


def find_attention_layers(model):
    for i, layer in enumerate(model.model.model.layers):
        yield i, layer.self_attn


def collect_kvs_for_basis(model, input_ids):
    kv_store = {}
    hooks = []
    for layer_idx, attn in find_attention_layers(model):
        for kv_type, proj_name in [('K', 'k_proj'), ('V', 'v_proj')]:
            def make_hook(li, kvt, nh=N_KV_HEADS, dh=D_HEAD):
                def hook(module, inp, out):
                    x = out.detach().cpu().float()
                    x = x.reshape(x.shape[0], x.shape[1], nh, dh)[0]
                    for h in range(nh):
                        key = (li, h)
                        if key not in kv_store:
                            kv_store[key] = {'K': [], 'V': []}
                        kv_store[key][kvt].append(x[:, h, :].numpy())
                return hook
            hooks.append(getattr(attn, proj_name).register_forward_hook(
                make_hook(layer_idx, kv_type)))
    with torch.no_grad():
        model(input_ids=input_ids)
    for h in hooks:
        h.remove()
    return {k: {kv: np.concatenate(v, axis=0) for kv, v in d.items()}
            for k, d in kv_store.items()}


def chunked_cross_entropy(model, input_ids, chunk_size=512):
    transformer_body = model.model.model
    lm_head = model.model.lm_head
    with torch.no_grad():
        outputs = transformer_body(input_ids=input_ids[:, :-1])
        hidden = outputs.last_hidden_state
    labels = input_ids[:, 1:].reshape(-1)
    total_loss = 0.0
    n_tok = 0
    with torch.no_grad():
        for start in range(0, hidden.shape[1], chunk_size):
            end = min(start + chunk_size, hidden.shape[1])
            chunk_logits = lm_head(hidden[:, start:end, :])
            chunk_labels = labels[start:end]
            loss = torch.nn.functional.cross_entropy(
                chunk_logits.reshape(-1, chunk_logits.size(-1)), chunk_labels)
            total_loss += float(loss) * (end - start)
            n_tok += (end - start)
            del chunk_logits, chunk_labels, loss
            torch.cuda.empty_cache()
    del hidden
    torch.cuda.empty_cache()
    return total_loss / n_tok


def eval_with_k_hook(model, input_ids, hook_fn):
    hooks = []
    for layer_idx, attn in find_attention_layers(model):
        def make_hook(li=layer_idx):
            def hook(module, inp, out):
                dev, dty = out.device, out.dtype
                x = out.detach().cpu().float()
                b, s, _ = x.shape
                x = x.reshape(b, s, N_KV_HEADS, D_HEAD)
                for h in range(N_KV_HEADS):
                    xh = x[0, :, h, :].numpy()
                    xh_q = hook_fn(li, h, xh)
                    x[0, :, h, :] = torch.from_numpy(xh_q)
                return x.reshape(b, s, N_KV_HEADS * D_HEAD).to(dty).to(dev)
            return hook
        hooks.append(attn.k_proj.register_forward_hook(make_hook()))
    loss = chunked_cross_entropy(model, input_ids)
    for h in hooks:
        h.remove()
    return float(np.exp(loss))


# ── Quantization ──────────────────────────────────────────────────────────────

def uniform_quant_dequant(x_np, n_bits=4):
    """Per-column min-max uniform quantization."""
    x = x_np.copy().astype(np.float32)
    n_levels = 2 ** n_bits
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    scale = (x_max - x_min) / (n_levels - 1)
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.clip(np.round((x - x_min) / scale), 0, n_levels - 1)
    return (q * scale + x_min).astype(np.float32)


# ── Compression methods ───────────────────────────────────────────────────────

def compress_plain_4bit(x, n_bits=N_BITS):
    """[A] No PCA, no rotation. Uniform quant of raw K."""
    return uniform_quant_dequant(x, n_bits=n_bits)


def compress_raw_rotation(x, R_full, n_bits=N_BITS):
    """[B] Full d×d random rotation of raw K → quantize all d dims → rotate back.
    Fair comparison at k=d: same number of stored dims as plain_4bit, different
    quantization grid. Isolates rotation benefit without PCA or truncation."""
    rotated = x @ R_full.T                          # (T, d)
    quant   = uniform_quant_dequant(rotated, n_bits) # quantize ALL d dims
    return quant @ R_full                            # rotate back  (T, d)


def compress_pca_only(x, k, U_k, mean_k, n_bits=N_BITS):
    """[C/E] PCA project to k dims → quantize → unproject. NO rotation.
    At k=d: pure coordinate change, no information loss.
    At k<d: truncation — last (d-k) PCA components discarded."""
    xc   = (x - mean_k) @ U_k[:, :k]               # (T, k)
    xc_q = uniform_quant_dequant(xc, n_bits)
    return xc_q @ U_k[:, :k].T + mean_k             # (T, d)


def compress_subrotq(x, k, U_k, mean_k, R_k, n_bits=N_BITS):
    """[D/F] Full SubRotQ: PCA to k dims + k×k rotation + quantize.
    R_k is a k×k orthogonal matrix. Same stored dims as pca_only."""
    xc      = (x - mean_k) @ U_k[:, :k]             # (T, k)
    rotated = xc @ R_k.T                             # (T, k)
    quant   = uniform_quant_dequant(rotated, n_bits)
    unrot   = quant @ R_k                            # (T, k)
    return unrot @ U_k[:, :k].T + mean_k             # (T, d)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    t0 = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)

    csv_path    = RESULTS_DIR / "exp37_rotation_ablation_v2.csv"
    report_path = RESULTS_DIR / "REPORT-37-rotation-ablation-v2.md"
    fieldnames  = ["label", "k", "bits", "ppl", "rel_ppl", "note"]

    print("=== Task F1 v2: Rotation Ablation (corrected) ===")
    print(f"Loading model {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    model.eval()

    print("Loading WikiText-2...")
    calib_ids = get_wikitext2_tokens(tokenizer, "train", CALIB_TOKENS, 'cuda')
    eval_ids  = get_wikitext2_tokens(tokenizer, "test",  EVAL_TOKENS,  'cuda')

    # Baseline
    print("\nBaseline PPL...")
    baseline_ppl = float(np.exp(chunked_cross_entropy(model, eval_ids)))
    print(f"  PPL = {baseline_ppl:.4f}")
    assert 3.0 <= baseline_ppl <= 10.0, f"SANITY FAIL: {baseline_ppl:.4f}"
    print("  SANITY OK")

    rows = [{"label": "baseline", "k": D_HEAD, "bits": 16,
             "ppl": round(baseline_ppl, 4), "rel_ppl": 1.0,
             "note": "fp16 KV cache, no compression"}]

    # Calibration — collect PCA bases
    print("\nCollecting KV basis...")
    kvs = collect_kvs_for_basis(model, calib_ids)
    print(f"  {len(kvs)} (layer, head) pairs")

    bases = {}   # k → {(li, hi): {'U_K': ..., 'mean_K': ...}}
    R     = {}   # k → k×k rotation matrix
    for k in [112, 128]:
        print(f"  Fitting k={k}...", end="", flush=True)
        bases[k] = {}
        for (li, hi), kv in kvs.items():
            U_k, mean_k = fit_pca(kv['K'], k)
            bases[k][(li, hi)] = {'U_K': U_k, 'mean_K': mean_k}
        R[k] = random_rotation_matrix(k, seed=SEED)
        print(" done")

    # Full d×d rotation matrix (for raw_rotation condition)
    R_full = random_rotation_matrix(D_HEAD, seed=SEED)

    def run(label, k, note, fn):
        print(f"\n  [{label}] {note}...")
        ppl = eval_with_k_hook(model, eval_ids, fn)
        rel = round(ppl / baseline_ppl, 4)
        ppl = round(ppl, 4)
        print(f"    PPL={ppl:.4f}  rel={rel:.4f}")
        rows.append({"label": label, "k": k, "bits": N_BITS,
                     "ppl": ppl, "rel_ppl": rel, "note": note})

    # ── k=128 conditions (no truncation) ────────────────────────────────────
    print("\n--- k=128 conditions (no truncation, k=d=128) ---")

    run("plain_4bit", D_HEAD,
        "uniform quant raw K, no PCA, no rotation",
        lambda li, hi, x: compress_plain_4bit(x))

    run("raw_rotation_128", D_HEAD,
        "full d×d rotation of raw K + quant all d dims + rotate back",
        lambda li, hi, x: compress_raw_rotation(x, R_full))

    run("pca_only_128", 128,
        "PCA k=128 (no truncation) → quant → unproject, no rotation",
        lambda li, hi, x: compress_pca_only(
            x, 128, bases[128][(li, hi)]['U_K'], bases[128][(li, hi)]['mean_K']))

    run("subrotq_128", 128,
        "PCA k=128 + 128×128 rotation + quant (full SubRotQ, no truncation)",
        lambda li, hi, x: compress_subrotq(
            x, 128, bases[128][(li, hi)]['U_K'], bases[128][(li, hi)]['mean_K'], R[128]))

    # ── k=112 conditions (truncation regime) ────────────────────────────────
    print("\n--- k=112 conditions (truncation: discard 16 of 128 dims) ---")

    run("pca_only_112", 112,
        "PCA k=112 (truncate 16 dims) → quant → unproject, no rotation",
        lambda li, hi, x: compress_pca_only(
            x, 112, bases[112][(li, hi)]['U_K'], bases[112][(li, hi)]['mean_K']))

    run("subrotq_112", 112,
        "PCA k=112 + 112×112 rotation + quant (full SubRotQ, truncated)",
        lambda li, hi, x: compress_subrotq(
            x, 112, bases[112][(li, hi)]['U_K'], bases[112][(li, hi)]['mean_K'], R[112]))

    # ── Write CSV ────────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # ── Report ───────────────────────────────────────────────────────────────
    wall = (time.time() - t0) / 60

    def get(lbl): return next((r for r in rows if r['label'] == lbl), None)

    with open(report_path, "w") as f:
        f.write("# Task F1 v2: Rotation Ablation — Qwen3-14B WikiText-2\n\n")
        f.write(f"Baseline PPL: {baseline_ppl:.4f} | Eval tokens: {EVAL_TOKENS} | "
                f"Wall: {wall:.1f} min\n\n")
        f.write("## Results\n\n")
        f.write("| Label | k | bits | PPL | rel PPL | Note |\n")
        f.write("|-------|---|------|-----|---------|------|\n")
        for r in rows:
            f.write(f"| {r['label']} | {r['k']} | {r['bits']} | "
                    f"{r['ppl']} | {r['rel_ppl']} | {r['note']} |\n")

        # 2×2 at k=128
        f.write("\n## 2×2 Ablation at k=128 (no truncation)\n\n")
        f.write("| | No Rotation | With Rotation |\n")
        f.write("|---|---|---|\n")
        pf  = get("plain_4bit");       rr = get("raw_rotation_128")
        po  = get("pca_only_128");     sr = get("subrotq_128")
        f.write(f"| **No PCA (raw K)** | {pf['rel_ppl'] if pf else '?'}× "
                f"| {rr['rel_ppl'] if rr else '?'}× |\n")
        f.write(f"| **PCA k=128** | {po['rel_ppl'] if po else '?'}× "
                f"| {sr['rel_ppl'] if sr else '?'}× |\n")

        f.write("\n## Rotation effect at k=112 (truncation regime)\n\n")
        po112 = get("pca_only_112");  sr112 = get("subrotq_112")
        if po112 and sr112:
            diff = round(float(sr112['rel_ppl']) - float(po112['rel_ppl']), 4)
            f.write(f"| pca_only_112 | {po112['rel_ppl']}× |\n")
            f.write(f"| subrotq_112  | {sr112['rel_ppl']}× |\n")
            f.write(f"\nRotation effect at k=112: {diff:+.4f}× "
                    f"({'helps' if diff < 0 else 'hurts or negligible'})\n")

        f.write("\n## Notes\n\n")
        f.write("- All differences <0.03× should be treated with caution at 2K eval tokens.\n")
        f.write("- 'rotation_only at k=112' removed (v1 bug: stored all d=128 dims, not a valid truncation ablation).\n")
        f.write("- raw_rotation_128 uses a full d×d rotation of raw K — valid comparison at k=d only.\n")

    print(f"\nCSV:    {csv_path}")
    print(f"Report: {report_path}")
    print(f"Wall time: {wall:.1f} min")
    print("=== Done ===")


if __name__ == "__main__":
    main()
