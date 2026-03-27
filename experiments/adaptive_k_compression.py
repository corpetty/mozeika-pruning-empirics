"""
adaptive_k_compression.py — Compare fixed-k vs adaptive-k (eff_rank_90) subspace compression.

Instead of using a fixed k for all layers/heads, use the per-head effective rank at 90%
variance as the subspace dimension. This allocates more bits per dimension in low-rank heads
and fewer in high-rank heads, while keeping the total bit budget constant.

Usage:
    /home/petty/torch-env/bin/python3 experiments/adaptive_k_compression.py
"""

import sys
import csv
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compress import (
    polar_quantize, subspace_polar_quantize, fit_pca,
    attention_score_distortion, random_rotation_matrix
)
from compress_standalone import load_analysis, generate_synthetic_kvs_from_spectrum

np.random.seed(42)

D_HEAD = 128
T = 512  # synthetic batch size


def run_adaptive_comparison(analysis: dict) -> list:
    rows = []
    layers = sorted(analysis.keys())

    for layer_idx in layers:
        for head_idx in sorted(analysis[layer_idx].keys()):
            hdata = analysis[layer_idx][head_idx]
            sk = hdata.get('sk', None)
            if sk is None:
                continue

            erk90 = int(hdata.get('erk90', D_HEAD))
            erk90 = max(4, min(erk90, D_HEAD - 1))  # clamp to valid range

            # Generate synthetic K vectors with measured spectrum
            K = generate_synthetic_kvs_from_spectrum(sk, T, D_HEAD, seed=layer_idx * 100 + head_idx)
            Q = generate_synthetic_kvs_from_spectrum(sk, T, D_HEAD, seed=layer_idx * 100 + head_idx + 1)

            T_cal = T // 2
            K_cal = K[:T_cal]
            K_test = K[T_cal:]
            Q_test = Q[T_cal:]

            # Two bit budgets: 256 bits/vector (2-bit full-dim) and 512 bits/vector (4-bit full-dim)
            for budget_label, full_bits_per_dim in [("2bit_budget", 2), ("4bit_budget", 4)]:
                total_budget = D_HEAD * full_bits_per_dim  # 256 or 512

                # --- Baseline: full_dim at this bit rate ---
                R_full = random_rotation_matrix(D_HEAD, seed=0)
                K_full_q = polar_quantize(K_test, full_bits_per_dim, R_full)
                d_full = attention_score_distortion(Q_test, K_test, K_full_q)

                rows.append({
                    'layer': layer_idx, 'head': head_idx,
                    'method': 'full_dim', 'k': D_HEAD,
                    'n_bits_per_dim': full_bits_per_dim,
                    'bits_per_vector': total_budget,
                    'budget': budget_label,
                    'eff_rank_90': erk90,
                    **{f'K_{m}': v for m, v in d_full.items()},
                })

                # --- Fixed k=64 subspace (the best fixed k from exp 0) ---
                k_fixed = 64
                n_bits_fixed = max(1, round(total_budget / k_fixed))
                U_fixed, mean_fixed = fit_pca(K_cal, k_fixed)
                R_fixed = random_rotation_matrix(k_fixed, seed=1)
                K_fixed_q = subspace_polar_quantize(K_test, k_fixed, n_bits_fixed, U_fixed, mean_fixed, R_fixed)
                d_fixed = attention_score_distortion(Q_test, K_test, K_fixed_q)

                rows.append({
                    'layer': layer_idx, 'head': head_idx,
                    'method': 'subspace_k64_fixed', 'k': k_fixed,
                    'n_bits_per_dim': n_bits_fixed,
                    'bits_per_vector': n_bits_fixed * k_fixed,
                    'budget': budget_label,
                    'eff_rank_90': erk90,
                    **{f'K_{m}': v for m, v in d_fixed.items()},
                })

                # --- Adaptive k = eff_rank_90 ---
                k_adapt = erk90
                n_bits_adapt = max(1, round(total_budget / k_adapt))
                U_adapt, mean_adapt = fit_pca(K_cal, k_adapt)
                R_adapt = random_rotation_matrix(k_adapt, seed=1)
                K_adapt_q = subspace_polar_quantize(K_test, k_adapt, n_bits_adapt, U_adapt, mean_adapt, R_adapt)
                d_adapt = attention_score_distortion(Q_test, K_test, K_adapt_q)

                rows.append({
                    'layer': layer_idx, 'head': head_idx,
                    'method': 'subspace_adaptive', 'k': k_adapt,
                    'n_bits_per_dim': n_bits_adapt,
                    'bits_per_vector': n_bits_adapt * k_adapt,
                    'budget': budget_label,
                    'eff_rank_90': erk90,
                    **{f'K_{m}': v for m, v in d_adapt.items()},
                })

            if head_idx == 0:
                print(f"  Layer {layer_idx:2d}: eff_rank_90={erk90:3d}, "
                      f"2bit: full={rows[-6]['K_kl_divergence']:.6f} "
                      f"k64={rows[-5]['K_kl_divergence']:.6f} "
                      f"adaptive(k={erk90})={rows[-4]['K_kl_divergence']:.6f}")

    return rows


def print_summary(rows: list):
    print("\n" + "=" * 80)
    print("ADAPTIVE K COMPRESSION — SUMMARY")
    print("=" * 80)

    for budget in ["2bit_budget", "4bit_budget"]:
        budget_rows = [r for r in rows if r['budget'] == budget]
        print(f"\n--- {budget} ({256 if '2' in budget else 512} bits/vector) ---")

        for method in ['full_dim', 'subspace_k64_fixed', 'subspace_adaptive']:
            subset = [r for r in budget_rows if r['method'] == method]
            if not subset:
                continue
            kl_vals = [r['K_kl_divergence'] for r in subset]
            top1_vals = [r['K_top1_agreement'] for r in subset]
            k_vals = [r['k'] for r in subset]
            print(f"\n  {method}:")
            print(f"    Mean KL divergence: {np.mean(kl_vals):.6f}")
            print(f"    Mean top-1 agreement: {np.mean(top1_vals):.4f}")
            print(f"    k range: {min(k_vals)}–{max(k_vals)} (mean={np.mean(k_vals):.0f})")

            # Per layer range breakdown
            for label, lo, hi in [("Early L0-9", 0, 10), ("Mid L10-29", 10, 30), ("Late L30-39", 30, 40)]:
                sub = [r for r in subset if lo <= r['layer'] < hi]
                if sub:
                    print(f"    {label}: KL={np.mean([r['K_kl_divergence'] for r in sub]):.6f}, "
                          f"top1={np.mean([r['K_top1_agreement'] for r in sub]):.4f}")

        # Improvement ratio: adaptive vs fixed k=64
        fixed = {(r['layer'], r['head']): r for r in budget_rows if r['method'] == 'subspace_k64_fixed'}
        adaptive = {(r['layer'], r['head']): r for r in budget_rows if r['method'] == 'subspace_adaptive'}
        full = {(r['layer'], r['head']): r for r in budget_rows if r['method'] == 'full_dim'}

        if fixed and adaptive:
            ratios = []
            wins_adapt = 0
            for key in fixed:
                if key in adaptive and fixed[key]['K_kl_divergence'] > 0:
                    ratio = adaptive[key]['K_kl_divergence'] / fixed[key]['K_kl_divergence']
                    ratios.append(ratio)
                    if ratio < 1.0:
                        wins_adapt += 1
            print(f"\n  Adaptive vs fixed k=64:")
            print(f"    Mean KL ratio (adaptive/fixed): {np.mean(ratios):.4f}")
            print(f"    Adaptive wins: {wins_adapt}/{len(ratios)} heads ({100*wins_adapt/len(ratios):.0f}%)")

        # Show adaptive k distribution
        adapt_rows = [r for r in budget_rows if r['method'] == 'subspace_adaptive']
        if adapt_rows:
            k_vals = [r['k'] for r in adapt_rows]
            bits_vals = [r['n_bits_per_dim'] for r in adapt_rows]
            print(f"\n  Adaptive k distribution:")
            print(f"    k: min={min(k_vals)}, max={max(k_vals)}, mean={np.mean(k_vals):.1f}, median={np.median(k_vals):.0f}")
            print(f"    bits/dim: min={min(bits_vals)}, max={max(bits_vals)}, mean={np.mean(bits_vals):.1f}")


def main():
    import os
    os.chdir(Path(__file__).resolve().parent.parent)
    Path("results").mkdir(exist_ok=True)

    print("=" * 80)
    print("Adaptive k Compression Experiment")
    print("Comparing fixed k=64 vs adaptive k=eff_rank_90 subspace compression")
    print("=" * 80)

    analysis = load_analysis('results/analysis.npz')
    print(f"Loaded analysis: {len(analysis)} layers\n")

    rows = run_adaptive_comparison(analysis)

    # Save CSV
    with open('results/adaptive_k_distortion.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to results/adaptive_k_distortion.csv")

    print_summary(rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
