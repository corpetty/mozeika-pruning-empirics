"""
Experiment 28: Adaptive K-Scheduling with Multi-Seed Error Bars.

MOTIVATION
----------
Exp18 showed rank-proportional scheduling beats uniform k at all budgets, but ran
only a single calibration sample (War & Peace, seed=0). The result could be lucky
calibration data. This experiment repeats exp18's three best budget points
(mean_k ∈ {96, 104, 112}) with 5 different WikiText-2 calibration passages and
reports mean ± std for each policy/budget combination.

Design:
  - 5 seeds: different 2K-token calibration windows from WikiText-2 train split
  - 3 budget targets: mean_k ∈ {96, 104, 112}  (the "viable" range from exp18)
  - 2 policies per budget: uniform k, rank-proportional adaptive k
  - Evaluation: WikiText-2 test split (same eval set across seeds)
  - K candidates: {64, 80, 96, 112, 128}

Error bars: std across 5 calibration seeds (not resampling/bootstrap).

Usage:
    /home/petty/torch-env/bin/python3 experiments/exp28_adaptive_errorbars.py

Outputs:
    results/exp28_adaptive_errorbars.csv
    results/REPORT-28-adaptive-errorbars.md
"""

import sys
import csv
import json
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compress import subspace_polar_quantize, fit_pca
from collect import get_model_and_tokenizer, find_attention_layers

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME   = "Qwen/Qwen3-14B-AWQ"
CALIB_TOKENS = 2048
EVAL_TOKENS  = 4096
BITS         = 4
K_CANDIDATES = [64, 80, 96, 112, 128]
BUDGET_TARGETS = [96, 104, 112]
N_SEEDS      = 5

SENSITIVITY_CSV = RESULTS_DIR / "exp16_layer_sensitivity.csv"

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_wikitext2_tokens(tokenizer, split, n_tokens, offset_tokens=0):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, trust_remote_code=True)
    text = "\n\n".join(ds["text"])
    ids = tokenizer.encode(text)
    start = offset_tokens
    return ids[start: start + n_tokens]


def load_sensitivity():
    import csv as _csv
    rows = []
    with open(SENSITIVITY_CSV) as f:
        for r in _csv.DictReader(f):
            rows.append((int(r["layer_idx"]), float(r["ppl_delta"])))
    rows.sort(key=lambda x: x[0])
    return rows  # [(layer_idx, ppl_delta), ...]


def rank_proportional_assignment(sensitivities, budget_k):
    """Assign k per layer proportional to sensitivity rank, hitting ~budget_k mean."""
    n = len(sensitivities)
    order = sorted(range(n), key=lambda i: sensitivities[i][1], reverse=True)
    k_assign = {i: K_CANDIDATES[len(K_CANDIDATES)//2] for i in range(n)}
    # Scale: most sensitive → k=128, least → k=64, linearly
    for rank, layer_i in enumerate(order):
        frac = rank / (n - 1)  # 0 = most sensitive, 1 = least
        # Interpolate across K_CANDIDATES
        k_idx = round(frac * (len(K_CANDIDATES) - 1))
        k_assign[layer_i] = K_CANDIDATES[k_idx]

    # Adjust to meet budget
    current_mean = sum(k_assign.values()) / n
    diff = budget_k - current_mean
    if abs(diff) > 2:
        # Scale all k values toward budget
        for i in range(n):
            new_k = k_assign[i] + diff
            # Snap to nearest candidate
            best = min(K_CANDIDATES, key=lambda x: abs(x - new_k))
            k_assign[i] = best
    return k_assign


def collect_kvs_for_basis(model, input_ids, n_kv_heads=8, d_head=128):
    """
    Collect K and V projections for PCA basis fitting.
    Returns {(layer_idx, head_idx): {'K': (T, d_head), 'V': (T, d_head)}}

    This is the PROVEN pattern from exp24. Do NOT modify.
    """
    kv_store = {}
    hooks = []

    for layer_idx, attn in find_attention_layers(model):
        for kv_type, proj_name in [('K', 'k_proj'), ('V', 'v_proj')]:
            proj = getattr(attn, proj_name)
            def make_hook(li, kvt, nh, dh):
                def hook(module, inp, out):
                    x = out.detach().cpu().float()
                    x = x.reshape(x.shape[0], x.shape[1], nh, dh)[0]  # (T, nh, dh)
                    for h_idx in range(nh):
                        key = (li, h_idx)
                        if key not in kv_store:
                            kv_store[key] = {'K': [], 'V': []}
                        kv_store[key][kvt].append(x[:, h_idx, :].numpy())
                return hook
            hooks.append(proj.register_forward_hook(make_hook(layer_idx, kv_type, n_kv_heads, d_head)))

    with torch.no_grad():
        model(input_ids=input_ids)

    for h in hooks:
        h.remove()

    return {k: {kv: np.concatenate(v, axis=0) for kv, v in d.items()}
            for k, d in kv_store.items()}


def chunked_cross_entropy(model, tokenizer, token_ids, k_assign, bases, device, chunk=512):
    """Compute PPL with K-subspace compression hooks, chunked projection."""
    n_kv_heads = 8
    d_head = 128
    hooks = []

    for layer_idx, attn_mod in find_attention_layers(model):
        try:
            k_mod = attn_mod.k_proj
        except AttributeError:
            continue
        k_val = k_assign[layer_idx]
        if k_val >= d_head:
            continue

        def make_compress_hook(li, k):
            def hook_fn(module, inp, out):
                T = out.shape[1]
                flat = out[0].detach().float().cpu().numpy().reshape(T, n_kv_heads, d_head)
                compressed = np.zeros_like(flat)
                for hi in range(n_kv_heads):
                    key = (li, hi)
                    if key not in bases:
                        compressed[:, hi, :] = flat[:, hi, :]
                        continue
                    U, mean = bases[key]
                    for start in range(0, T, chunk):
                        end = min(start + chunk, T)
                        seg = flat[start:end, hi, :]
                        # Compress: project to k dims then reconstruct
                        centered = seg - mean
                        coords = centered @ U[:, :k]
                        reconstructed = coords @ U[:, :k].T + mean
                        compressed[start:end, hi, :] = reconstructed
                out_tensor = torch.tensor(
                    compressed.reshape(T, n_kv_heads * d_head),
                    dtype=out.dtype, device=out.device
                ).unsqueeze(0)
                return out_tensor
            return hook_fn

        hooks.append(k_mod.register_forward_hook(make_compress_hook(layer_idx, k_val)))

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits[0]  # (T, vocab)

    for h in hooks:
        h.remove()

    # PPL
    targets = input_ids[0, 1:]
    log_probs = torch.nn.functional.log_softmax(logits[:-1], dim=-1)
    nll = -log_probs[torch.arange(len(targets)), targets].mean()
    return float(torch.exp(nll).cpu())


def fit_bases(k_arrays, k_max=128):
    """Fit PCA bases for all (layer, head) pairs. k_arrays = {(li,hi): np.array(T,d)}."""
    bases = {}
    for key, X in k_arrays.items():
        if len(X) < k_max:
            continue
        X_centered = X - X.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        bases[key] = (Vt[:k_max].T, X.mean(0))  # (d, k), mean
    return bases


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("Loading model...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    device = next(model.parameters()).device
    n_layers = 40

    print("Loading sensitivity scores...")
    sensitivities = load_sensitivity()  # [(layer_idx, ppl_delta)]

    print("Loading evaluation tokens (WikiText-2 test)...")
    eval_tokens = get_wikitext2_tokens(tokenizer, "test", EVAL_TOKENS, offset_tokens=0)

    # Precompute uniform k assignments per budget
    uniform_assign = {}
    for bk in BUDGET_TARGETS:
        best_k = min(K_CANDIDATES, key=lambda x: abs(x - bk))
        uniform_assign[bk] = {i: best_k for i in range(n_layers)}

    # Precompute rank-proportional assignments per budget
    rank_assign = {bk: rank_proportional_assignment(sensitivities, bk) for bk in BUDGET_TARGETS}

    results = []
    csv_path = RESULTS_DIR / "exp28_adaptive_errorbars.csv"

    # Seed loop: different calibration windows
    for seed in range(N_SEEDS):
        calib_offset = seed * CALIB_TOKENS * 3  # non-overlapping windows
        print(f"\n=== Seed {seed} (calib offset={calib_offset}) ===")

        calib_tokens = get_wikitext2_tokens(tokenizer, "train", CALIB_TOKENS, offset_tokens=calib_offset)

        calib_tensor = torch.tensor([calib_tokens], dtype=torch.long, device=device)
        print("  Collecting KVs for basis fitting...")
        kvs = collect_kvs_for_basis(model, calib_tensor)
        bases = fit_bases({k: v['K'] for k, v in kvs.items()})  # K-only
        print(f"  Fitted {len(bases)} bases")

        # Baseline PPL (no hooks)
        baseline_assign = {i: 128 for i in range(n_layers)}  # k=128 = no truncation
        baseline_ppl = chunked_cross_entropy(model, tokenizer, eval_tokens, baseline_assign, {}, device)
        print(f"  Baseline PPL: {baseline_ppl:.4f}")

        for bk in BUDGET_TARGETS:
            for policy_name, assign in [("uniform", uniform_assign[bk]), ("rank_prop", rank_assign[bk])]:
                mean_k = sum(assign.values()) / len(assign)
                ppl = chunked_cross_entropy(model, tokenizer, eval_tokens, assign, bases, device)
                rel_ppl = ppl / baseline_ppl
                print(f"  budget={bk} policy={policy_name} mean_k={mean_k:.1f} ppl={ppl:.4f} rel={rel_ppl:.4f}")

                row = {
                    "seed": seed,
                    "budget_k": bk,
                    "policy": policy_name,
                    "mean_k": mean_k,
                    "baseline_ppl": baseline_ppl,
                    "ppl": ppl,
                    "rel_ppl": rel_ppl,
                }
                results.append(row)

                # Save incrementally
                fieldnames = list(row.keys())
                write_header = not csv_path.exists()
                with open(csv_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        w.writeheader()
                    w.writerow(row)

    # ── Report ────────────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(results)

    print("\n\nGenerating report...")
    lines = ["# Experiment 28: Adaptive Scheduling Error Bars\n",
             f"N_SEEDS={N_SEEDS}, budgets={BUDGET_TARGETS}, WikiText-2 calib/eval\n",
             "\n## Summary (mean ± std across seeds)\n",
             "| Budget k | Policy | Mean k | Rel PPL (mean) | Rel PPL (std) |",
             "|----------|--------|--------|----------------|---------------|"]
    for bk in BUDGET_TARGETS:
        for policy in ["uniform", "rank_prop"]:
            sub = df[(df.budget_k == bk) & (df.policy == policy)]
            mean_k = sub["mean_k"].mean()
            rel_mean = sub["rel_ppl"].mean()
            rel_std = sub["rel_ppl"].std()
            lines.append(f"| {bk} | {policy} | {mean_k:.1f} | {rel_mean:.4f} | ±{rel_std:.4f} |")

    lines += [
        "\n## Key Question",
        "Does rank-proportional scheduling consistently beat uniform k across calibration seeds,",
        "or was exp18's result a lucky draw?",
        "",
        "## Conclusion",
    ]
    # Auto-conclude
    rank_wins = 0
    total = 0
    for bk in BUDGET_TARGETS:
        u = df[(df.budget_k == bk) & (df.policy == "uniform")]["rel_ppl"].mean()
        r = df[(df.budget_k == bk) & (df.policy == "rank_prop")]["rel_ppl"].mean()
        if r < u:
            rank_wins += 1
        total += 1
    lines.append(f"Rank-proportional beats uniform in {rank_wins}/{total} budget points (mean across {N_SEEDS} seeds).")

    report = "\n".join(lines)
    (RESULTS_DIR / "REPORT-28-adaptive-errorbars.md").write_text(report)
    print(report)
    print(f"\nDone. Results: {csv_path}")


if __name__ == "__main__":
    main()
