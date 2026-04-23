"""
Task F1: Rotation ablation — does the random rotation actually help at k=128?

MOTIVATION
----------
SubRotQ applies a random orthogonal rotation R before uniform quantization.
The claim: rotation equalizes variance across dimensions, reducing max quant error.

But A1 showed: SubRotQ k=128 (0.981x) ≈ plain 4-bit (0.989x) on 2K tokens.
The gap is within noise. So we need a more definitive ablation:

  1. PCA → rotate R → quantize       (SubRotQ — standard)
  2. PCA → NO rotation → quantize    (ablation: does R matter at full rank?)
  3. No PCA → rotate R → quantize    (ablation: does PCA matter, or is R doing the work?)
  4. No PCA → no rotation → quantize (plain uniform — A1 baseline)

All tested at k=128/4-bit and k=112/4-bit.

Expected:
- If rotation matters: (1) << (2) in PPL
- If PCA matters: (1) << (3) in PPL
- If neither matters at k=128: all ≈ baseline

Output:
  results/exp36_rotation_ablation.csv
  results/REPORT-36-rotation-ablation.md
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
from compress import fit_pca, random_rotation_matrix, quantize_uniform as _q_uniform

MODEL_NAME   = "Qwen/Qwen3-14B-AWQ"
RESULTS_DIR  = Path("results")
CALIB_TOKENS = 2048
EVAL_TOKENS  = 2048
N_KV_HEADS   = 8
D_HEAD       = 128
N_LAYERS     = 40
K_VALUES     = [112, 128]
N_BITS       = 4


def get_wikitext2_tokens(tokenizer, split, n_tokens, device):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    text = "\n".join(line for line in text.split("\n") if line.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if ids.shape[1] < n_tokens + 1:
        raise ValueError(f"Only {ids.shape[1]} tokens")
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


def uniform_quant_dequant(x_np, n_bits=4):
    x = x_np.copy().astype(np.float32)
    n_levels = 2 ** n_bits
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    scale = (x_max - x_min) / (n_levels - 1)
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.clip(np.round((x - x_min) / scale), 0, n_levels - 1)
    return (q * scale + x_min).astype(np.float32)


def compress_pca_only(x, k, U_k, mean_k, n_bits):
    """PCA project → quantize → unproject. NO rotation."""
    xc = x - mean_k                       # (T, d)
    proj = xc @ U_k                       # (T, k)
    proj_q = uniform_quant_dequant(proj[:, :k], n_bits=n_bits)
    # Pad to k if needed (for k < d case)
    return proj_q @ U_k[:, :k].T + mean_k  # (T, d)


def compress_rotation_only(x, k, R, n_bits):
    """Rotate raw K → quantize k dims → rotate back. NO PCA."""
    # R is (k, k) but x is (T, d=128); we need a d×d rotation
    # Use a full-rank R_full derived from R (pad identity)
    d = x.shape[1]
    R_full = np.eye(d, dtype=np.float32)
    R_full[:k, :k] = R  # embed k×k rotation in top-left corner
    rotated = x @ R_full.T                # (T, d)
    quant_k = uniform_quant_dequant(rotated[:, :k], n_bits=n_bits)  # quantize top k
    rotated_q = rotated.copy()
    rotated_q[:, :k] = quant_k
    return rotated_q @ R_full             # rotate back


def compress_subrotq(x, k, U_k, mean_k, R, n_bits):
    """Full SubRotQ: PCA + rotation + quant."""
    xc = x - mean_k
    proj = xc @ U_k[:, :k]               # (T, k)
    rotated = proj @ R.T                   # (T, k)
    quant = uniform_quant_dequant(rotated, n_bits=n_bits)
    unrotated = quant @ R                  # (T, k)
    return unrotated @ U_k[:, :k].T + mean_k  # (T, d)


def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    t0 = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)

    csv_path    = RESULTS_DIR / "exp36_rotation_ablation.csv"
    report_path = RESULTS_DIR / "REPORT-36-rotation-ablation.md"
    fieldnames  = ["method", "k", "bits", "ppl", "rel_ppl", "note"]

    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add((row["method"], int(row["k"])))
        print(f"Resuming: {len(done)} configs done")

    print(f"=== Task F1: Rotation Ablation ===")
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
    assert 3.0 <= baseline_ppl <= 10.0, f"SANITY FAIL: {baseline_ppl}"
    print("  SANITY OK")

    rows = [{"method": "baseline", "k": D_HEAD, "bits": 16,
             "ppl": round(baseline_ppl, 4), "rel_ppl": 1.0, "note": "no compression"}]

    # Calibration
    print("\nCollecting KV basis...")
    kvs = collect_kvs_for_basis(model, calib_ids)
    print(f"  {len(kvs)} pairs")

    bases_by_k = {}
    R_by_k = {}
    for k in K_VALUES:
        print(f"  Fitting k={k}...", end="", flush=True)
        bases_by_k[k] = {}
        for (li, hi), kv in kvs.items():
            U_k, mean_k = fit_pca(kv['K'], k)
            bases_by_k[k][(li, hi)] = {'U_K': U_k, 'mean_K': mean_k}
        R_by_k[k] = random_rotation_matrix(k, seed=0)
        print(" done")

    # Plain 4-bit (no PCA, no rotation)
    if ("plain_4bit", D_HEAD) not in done:
        print("\n[A] Plain 4-bit (no PCA, no rotation)...")
        def plain_fn(li, hi, x): return uniform_quant_dequant(x, n_bits=N_BITS)
        ppl = eval_with_k_hook(model, eval_ids, plain_fn)
        rel = ppl / baseline_ppl
        print(f"  PPL={ppl:.4f}  rel={rel:.4f}")
        rows.append({"method": "plain_4bit", "k": D_HEAD, "bits": N_BITS,
                     "ppl": round(ppl, 4), "rel_ppl": round(rel, 4),
                     "note": "no PCA, no rotation"})

    for k in K_VALUES:
        # PCA only (no rotation)
        if ("pca_only", k) not in done:
            print(f"\n[B] PCA only (no rotation) k={k}...")
            def pca_fn(li, hi, x, _k=k):
                b = bases_by_k[_k][(li, hi)]
                return compress_pca_only(x, _k, b['U_K'], b['mean_K'], N_BITS)
            ppl = eval_with_k_hook(model, eval_ids, pca_fn)
            rel = ppl / baseline_ppl
            print(f"  PPL={ppl:.4f}  rel={rel:.4f}")
            rows.append({"method": "pca_only", "k": k, "bits": N_BITS,
                         "ppl": round(ppl, 4), "rel_ppl": round(rel, 4),
                         "note": f"PCA k={k} → quantize, no rotation"})

        # Rotation only (no PCA)
        if ("rotation_only", k) not in done:
            print(f"\n[C] Rotation only (no PCA) k={k}...")
            def rot_fn(li, hi, x, _k=k):
                return compress_rotation_only(x, _k, R_by_k[_k], N_BITS)
            ppl = eval_with_k_hook(model, eval_ids, rot_fn)
            rel = ppl / baseline_ppl
            print(f"  PPL={ppl:.4f}  rel={rel:.4f}")
            rows.append({"method": "rotation_only", "k": k, "bits": N_BITS,
                         "ppl": round(ppl, 4), "rel_ppl": round(rel, 4),
                         "note": f"rotate raw K (embed k×k in identity), no PCA"})

        # SubRotQ (PCA + rotation)
        if ("subrotq", k) not in done:
            print(f"\n[D] SubRotQ (PCA + rotation) k={k}...")
            def sr_fn(li, hi, x, _k=k):
                b = bases_by_k[_k][(li, hi)]
                return compress_subrotq(x, _k, b['U_K'], b['mean_K'], R_by_k[_k], N_BITS)
            ppl = eval_with_k_hook(model, eval_ids, sr_fn)
            rel = ppl / baseline_ppl
            print(f"  PPL={ppl:.4f}  rel={rel:.4f}")
            rows.append({"method": "subrotq", "k": k, "bits": N_BITS,
                         "ppl": round(ppl, 4), "rel_ppl": round(rel, 4),
                         "note": f"PCA k={k} + random rotation + 4-bit"})

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Report
    wall = (time.time() - t0) / 60

    def get(m, k): return next((r for r in rows if r['method'] == m and r.get('k', D_HEAD) == k), None)

    with open(report_path, "w") as f:
        f.write("# Task F1: Rotation Ablation — Qwen3-14B WikiText-2\n\n")
        f.write(f"Baseline PPL: {baseline_ppl:.4f} | Wall: {wall:.1f} min\n\n")
        f.write("| Method | k | bits | PPL | rel PPL | Note |\n")
        f.write("|--------|---|------|-----|---------|------|\n")
        for r in rows:
            f.write(f"| {r['method']} | {r.get('k',D_HEAD)} | {r.get('bits',16)} | {r['ppl']} | {r['rel_ppl']} | {r.get('note','')} |\n")

        f.write("\n## 2×2 Ablation Summary (k=128)\n\n")
        f.write("| | No Rotation | With Rotation |\n")
        f.write("|---|---|---|\n")
        pca_no  = get("pca_only", 128)
        pca_yes = get("subrotq", 128)
        rot_no  = get("plain_4bit", D_HEAD)
        rot_yes = get("rotation_only", 128)
        f.write(f"| **No PCA** | {rot_no['rel_ppl'] if rot_no else '?'}× | {rot_yes['rel_ppl'] if rot_yes else '?'}× |\n")
        f.write(f"| **PCA k=128** | {pca_no['rel_ppl'] if pca_no else '?'}× | {pca_yes['rel_ppl'] if pca_yes else '?'}× |\n")

        f.write("\n## Interpretation\n\n")
        if pca_yes and pca_no:
            r_diff = float(pca_yes['rel_ppl']) - float(pca_no['rel_ppl'])
            f.write(f"SubRotQ vs PCA-only at k=128: {r_diff:+.4f}x. ")
            if abs(r_diff) < 0.01:
                f.write("**Rotation adds negligible benefit at full rank (k=d).**\n")
            else:
                f.write(f"**Rotation {'helps' if r_diff < 0 else 'hurts'} at full rank.**\n")
        if pca_yes and rot_no:
            p_diff = float(pca_yes['rel_ppl']) - float(rot_no['rel_ppl'])
            f.write(f"SubRotQ vs plain 4-bit: {p_diff:+.4f}x. ")
            if abs(p_diff) < 0.01:
                f.write("**Full pipeline = plain quant at full rank — SubRotQ's value is in truncation.**\n")

    print(f"\nCSV: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Wall time: {wall:.1f} min")
    print("=== Done ===")


if __name__ == "__main__":
    main()
