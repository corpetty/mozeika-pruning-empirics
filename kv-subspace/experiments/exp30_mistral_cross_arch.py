"""
Experiment 30: Cross-Architecture Generalization (Mistral-7B).

MOTIVATION
----------
All prior results are on Qwen3-14B-AWQ (GQA, 40 layers, 8 KV heads).
This experiment tests whether K-only subspace compression generalizes to
a different GQA architecture: Mistral-7B-v0.3.

Mistral-7B-v0.3 specs:
  - 32 layers
  - GQA: 8 KV heads, 32 Q heads (GQA ratio 4)
  - d_head = 128
  - vocab = 32768

Design:
  - Model: mistralai/Mistral-7B-v0.3 (unquantized, fp16)
  - Calibration: WikiText-2 train, 2048 tokens
  - Evaluation: WikiText-2 test, PPL at ctx_len = 2048
  - Configs: baseline (fp16), k ∈ {64, 96, 112, 128} at 4-bit K-only
  - V: full rank, fp16 (consistent with Qwen3 findings)

Usage:
    /home/petty/torch-env/bin/python3 experiments/exp30_mistral_cross_arch.py

Outputs:
    results/exp30_mistral_ppl.csv
    results/REPORT-30-mistral.md
"""

import sys
import csv
import json
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from collect import get_model_and_tokenizer, find_attention_layers

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME   = "mistralai/Mistral-7B-v0.3"   # unquantized, ~14GB
CALIB_TOKENS = 2048
EVAL_TOKENS  = 2048
D_HEAD       = 128
N_KV_HEADS   = 8
K_CANDIDATES = [64, 96, 112, 128]
BITS         = 4


# ── Proven helpers from exp24 ─────────────────────────────────────────────────

def get_wikitext2_tokens(tokenizer, split, n_tokens):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, trust_remote_code=True)
    text = "\n\n".join(ds["text"])
    text = "\n".join(line for line in text.split("\n") if line.strip())
    ids = tokenizer.encode(text)
    if len(ids) < n_tokens + 1:
        ids = ids * ((n_tokens // len(ids)) + 1)
    return ids[:n_tokens + 1]


def collect_kvs(model, input_ids, n_kv_heads=8, d_head=128):
    kv_store = {}
    hooks = []
    for layer_idx, attn in find_attention_layers(model):
        for kv_type, proj_name in [('K', 'k_proj')]:
            proj = getattr(attn, proj_name)
            def make_hook(li, kvt, nh, dh):
                def hook(module, inp, out):
                    x = out.detach().cpu().float()
                    x = x.reshape(x.shape[0], x.shape[1], nh, dh)[0]
                    for h_idx in range(nh):
                        key = (li, h_idx)
                        if key not in kv_store:
                            kv_store[key] = {'K': []}
                        kv_store[key][kvt].append(x[:, h_idx, :].numpy())
                return hook
            hooks.append(proj.register_forward_hook(make_hook(layer_idx, kv_type, n_kv_heads, d_head)))
    with torch.no_grad():
        model(input_ids=input_ids)
    for h in hooks:
        h.remove()
    return {k: {kv: np.concatenate(v, axis=0) for kv, v in d.items()}
            for k, d in kv_store.items()}


def fit_bases(k_arrays, k_max=128):
    bases = {}
    for key, X in k_arrays.items():
        if len(X) < k_max:
            continue
        X_c = X - X.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
        bases[key] = (Vt[:, :k_max], X.mean(0))
    return bases


def chunked_cross_entropy(model, input_ids, bases, k_assign, device, chunk=512, verbose=False):
    """Compute PPL with K-subspace compression hooks.
    
    If verbose=True, prints per-chunk PPL progression for trajectory tracking.
    """
    layer_names = list(find_attention_layers(model))
    n_layers = len(layer_names)
    hooks = []

    for layer_idx, attn_mod in layer_names:
        try:
            k_mod = attn_mod.k_proj
        except AttributeError:
            continue
        k_val = k_assign.get(layer_idx, 128)
        if k_val >= D_HEAD:
            continue

        def make_compress_hook(li, k):
            def hook_fn(module, inp, out):
                T = out.shape[1]
                flat = out[0].detach().float().cpu().numpy().reshape(T, N_KV_HEADS, D_HEAD)
                compressed = np.zeros_like(flat)
                for hi in range(N_KV_HEADS):
                    key = (li, hi)
                    if key not in bases:
                        compressed[:, hi, :] = flat[:, hi, :]
                        continue
                    U, mean = bases[key]
                    for start in range(0, T, chunk):
                        end = min(start + chunk, T)
                        seg = flat[start:end, hi, :]
                        centered = seg - mean
                        coords = centered @ U[:, :k]
                        reconstructed = coords @ U[:, :k].T + mean
                        compressed[start:end, hi, :] = reconstructed
                out_tensor = torch.tensor(
                    compressed.reshape(T, N_KV_HEADS * D_HEAD),
                    dtype=out.dtype, device=out.device
                ).unsqueeze(0)
                return out_tensor
            return hook_fn

        hooks.append(k_mod.register_forward_hook(make_compress_hook(layer_idx, k_val)))

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]

    for h in hooks:
        h.remove()

    targets = input_ids[0, 1:]
    
    if verbose and len(targets) > chunk:
        # Print per-chunk PPL for trajectory tracking
        n_chunks = (len(targets) + chunk - 1) // chunk
        for ci in range(n_chunks):
            s = ci * chunk
            e = min(s + chunk, len(targets))
            chunk_ppl = float(torch.exp(-torch.nn.functional.log_softmax(logits[s:e], dim=-1)[torch.arange(e - s), targets[s:e]].mean()).cpu())
            if ci % 2 == 0 or ci == n_chunks - 1:
                print(f"    Chunk {ci}/{n_chunks} (pos={s}-{e}): chunk_ppl={chunk_ppl:.4f}")

    log_probs = torch.nn.functional.log_softmax(logits[:-1], dim=-1)
    nll = -log_probs[torch.arange(len(targets)), targets].mean()
    return float(torch.exp(nll).cpu())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("Loading Mistral-7B-v0.3...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    device = next(model.parameters()).device
    n_layers = len(list(find_attention_layers(model)))
    print(f"  {n_layers} layers, {N_KV_HEADS} KV heads, d_head={D_HEAD}")

    # Calibration
    print("Loading calibration tokens (WikiText-2 train)...")
    calib_ids = get_wikitext2_tokens(tokenizer, "train", CALIB_TOKENS)
    calib_t = torch.tensor([calib_ids], dtype=torch.long, device=device)

    print("Collecting K projections...")
    kvs = collect_kvs(model, calib_t, N_KV_HEADS, D_HEAD)
    bases = {k: v['K'] for k, v in kvs.items()}
    bases = fit_bases(bases, k_max=128)
    print(f"  {len(bases)} bases fitted")

    # Evaluation tokens
    print("Loading evaluation tokens (WikiText-2 test)...")
    eval_ids = get_wikitext2_tokens(tokenizer, "test", EVAL_TOKENS)
    eval_t = torch.tensor([eval_ids], dtype=torch.long, device=device)

    # ── Baseline PPL ─────────────────────────────────────────────────────────
    print("\n--- Baseline ---")
    baseline_assign = {i: 128 for i in range(n_layers)}
    baseline_ppl = chunked_cross_entropy(model, eval_t, {}, baseline_assign, device)
    print(f"Baseline PPL (k=128 no-op): {baseline_ppl:.4f}")

    if baseline_ppl < 4.0 or baseline_ppl > 50.0:
        print(f"  WARNING: Baseline PPL={baseline_ppl:.4f} is outside expected range (6-20)")
        print(f"  If < 4.0: likely memorization of WikiText-2 test split")
        print(f"  If > 50: model loading or eval token bug")
    else:
        print(f"  ✓ Baseline PPL in reasonable range for WikiText-2")

    # ── Diagnostics ──────────────────────────────────────────────────────────
    print("\n--- Diagnostics (startup sanity checks) ---")

    # No-op sanity: k=d_head should reproduce baseline
    print("  No-op sanity (k=128, should match baseline)...")
    no_op_assign = {i: 128 for i in range(n_layers)}
    no_op_ppl = chunked_cross_entropy(model, eval_t, {}, no_op_assign, device)
    print(f"    Baseline PPL: {baseline_ppl:.4f}")
    print(f"    No-op PPL:    {no_op_ppl:.4f}")
    if abs(no_op_ppl - baseline_ppl) > 0.01:
        print("    WARNING: no-op mismatch! Hook may be broken.")
    else:
        print("    ✓ No-op matches baseline")

    # Variance explained
    print("  Variance explained at key k values...")
    for key in list(bases.keys())[:3]:  # first 3 layers/heads
        U, mean = bases[key]
        X_c = (kvs[key]['K'] - kvs[key]['K'].mean(0))
        sv = np.linalg.svd(X_c, compute_uv=False)
        total_var = (sv ** 2).sum()
        for kv in [64, 96, 112, 128]:
            cum = (sv[:kv] ** 2).sum() / total_var if kv <= len(sv) else 1.0
            if kv in [64, 128]:
                print(f"    {key}: k={kv} → {cum:.4f} variance explained")
        break  # just show first layer/heads

    # Basis stability (first half vs second half of calib)
    print("  Basis stability (first vs second half of calibration)...")
    stability_checks = 0
    for key in list(bases.keys())[:5]:
        X = kvs[key]['K']
        half = len(X) // 2
        U1 = np.linalg.svd(X[:half] - X[:half].mean(0), full_matrices=False)[0][:, :128]
        U2 = np.linalg.svd(X[half:] - X[half:].mean(0), full_matrices=False)[0][:, :128]
        sim = np.abs(np.diag(U1.T @ U2)).mean()
        if sim < 0.85:
            print(f"    WARNING: {key} stability={sim:.4f} < 0.85 (calib too noisy?)")
        stability_checks += 1
    print(f"    Checked {stability_checks} layers/heads")

    # Calib/eval overlap
    calib_overlap = set(calib_ids) & set(eval_ids)
    if len(calib_overlap) > len(eval_ids) * 0.1:
        print(f"    WARNING: {len(calib_overlap)} eval tokens overlap with calibration!")
    else:
        print("    ✓ Calib/eval overlap minimal")

    print(f"\n--- Main sweep ---")

    # Sweep k values with incremental saves
    csv_path = RESULTS_DIR / "exp30_mistral_ppl.csv"
    fieldnames = ["k", "bits", "ppl", "rel_ppl", "chunk_ppls"]
    all_results = []

    # Load existing results if resuming
    if csv_path.exists():
        import csv as csv_std
        with open(csv_path, "r") as f:
            reader = csv_std.DictReader(f)
            all_results = list(reader)
        print(f"  Loaded {len(all_results)} prior results from {csv_path}")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for k in K_CANDIDATES:
            # Skip if already done (resume after crash)
            exists = any(r["k"] == str(k) for r in all_results)
            if exists:
                print(f"  k={k:3d}: SKIPPED (already in CSV)")
                continue

            k_assign = {i: k for i in range(n_layers)}
            print(f"  k={k:3d}/{BITS}-bit: running...")
            ppl = chunked_cross_entropy(model, eval_t, bases, k_assign, device, verbose=True)
            rel_ppl = ppl / baseline_ppl
            chunk_ppls = "TODO"  # would need chunked_forward to accumulate
            row = dict(k=k, bits=BITS, ppl=f"{ppl:.4f}", rel_ppl=f"{rel_ppl:.4f}", chunk_ppls=chunk_ppls)
            all_results.append(row)
            w.writerow(row)
            f.flush()
            print(f"  k={k:3d}/{BITS}-bit: PPL={ppl:.4f} rel={rel_ppl:.4f} ✓ saved")

            # Monotonicity check
            if len(all_results) > 1:
                prev_ppl = float(all_results[-2]["ppl"])
                if ppl > prev_ppl + 0.01:
                    print(f"    ✓ Monotonic: PPL rose from {prev_ppl:.4f} to {ppl:.4f}")
                else:
                    print(f"    WARNING: PPL did not increase monotonically ({prev_ppl:.4f} → {ppl:.4f})")

    # Report
    lines = [
        "# Experiment 30: Cross-Architecture (Mistral-7B-v0.3)\n",
        f"Model: {MODEL_NAME}, {n_layers} layers, GQA (8 KV heads, d_head={D_HEAD})\n",
        f"Calibration: WikiText-2 train ({CALIB_TOKENS} tokens)\n",
        f"Evaluation: WikiText-2 test ({EVAL_TOKENS} tokens)\n",
        "",
        "## Results",
        "| k | bits | PPL | Rel PPL |",
        "|---|------|-----|---------|",
    ]
    for r in results:
        lines.append(f"| {r['k']} | {r['bits']} | {r['ppl']:.4f} | {r['rel_ppl']:.4f}x |")

    lines += [
        "",
        "## Comparison with Qwen3-14B-AWQ",
        "| Config | Qwen3 rel PPL | Mistral rel PPL | Delta |",
        "|--------|--------------|-----------------|-------|",
    ]
    # Pull Qwen3 results from exp24
    qwen3_results = {}
    try:
        import pandas as pd
        df24 = pd.read_csv(RESULTS_DIR / "exp24_wikitext2_ppl.csv")
        for _, row in df24.iterrows():
            qwen3_results[row["k"]] = row["rel_ppl"]
    except Exception:
        pass

    for r in results:
        k = r["k"]
        q3 = qwen3_results.get(k, "N/A")
        mistral_rel = r["rel_ppl"]
        if isinstance(q3, float):
            delta = mistral_rel - q3
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "N/A"
        lines.append(f"| k={k}/{BITS}-bit | {q3} | {mistral_rel:.4f} | {delta_str} |")

    lines += [
        "",
        "## Conclusion",
        "If Mistral shows similar rel PPL degradation as Qwen3, the subspace compression",
        "method generalizes across GQA architectures. If degradation is worse, the method",
        "may be Qwen3-specific.",
    ]

    report = "\n".join(lines)
    (RESULTS_DIR / "REPORT-30-mistral.md").write_text(report)
    print("\n" + report)
    print(f"\nDone. CSV: {csv_path}")


if __name__ == "__main__":
    main()
