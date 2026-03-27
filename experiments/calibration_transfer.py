"""
calibration_transfer.py — Test whether a PCA basis calibrated on one text domain
transfers well to compressing KV vectors from a different domain.

Compares:
  (a) Oracle: PCA basis fitted on domain 2 test data itself
  (b) Transfer: PCA basis fitted on domain 1 (fiction), applied to domain 2 (factual)
  (c) Full-dim: No subspace projection, just PolarQuant at matched bits

Usage:
    /home/petty/torch-env/bin/python3 experiments/calibration_transfer.py
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
from collect import load_kvs

np.random.seed(42)

D_HEAD = 128
K_SUBSPACE = 64  # subspace dimension (the winner from earlier experiments)
BIT_BUDGETS = [2, 4]  # bits per dimension (full-dim equivalent)


def run_transfer_experiment(kvs1: dict, kvs2: dict) -> list:
    """Compare oracle, transfer, and full_dim compression on domain 2 data."""
    rows = []
    layers = sorted(set(kvs1.keys()) & set(kvs2.keys()))

    for layer_idx in layers:
        K1 = kvs1[layer_idx]['K']  # (T1, H, d) — domain 1
        K2 = kvs2[layer_idx]['K']  # (T2, H, d) — domain 2

        T1, H, D = K1.shape
        T2 = K2.shape[0]

        for head_idx in range(H):
            k1 = K1[:, head_idx, :]  # (T1, d) — domain 1, this head
            k2 = K2[:, head_idx, :]  # (T2, d) — domain 2, this head

            # Use first half of domain 2 as calibration, second half as test
            T2_cal = T2 // 2
            k2_cal = k2[:T2_cal]
            k2_test = k2[T2_cal:]

            # Generate Q vectors from domain 2 for attention score measurement
            # Use domain 2 K vectors as pseudo-queries (self-attention scenario)
            q2_test = k2[:T2_cal]  # use cal portion as queries against test portion

            for bits_per_dim in BIT_BUDGETS:
                total_budget = D * bits_per_dim
                n_bits_sub = max(1, round(total_budget / K_SUBSPACE))

                # --- (a) Oracle: PCA basis from domain 2 calibration set ---
                U_oracle, mean_oracle = fit_pca(k2_cal, K_SUBSPACE)
                R_sub = random_rotation_matrix(K_SUBSPACE, seed=1)
                k2_oracle_q = subspace_polar_quantize(
                    k2_test, K_SUBSPACE, n_bits_sub, U_oracle, mean_oracle, R_sub
                )
                d_oracle = attention_score_distortion(q2_test, k2_test, k2_oracle_q)

                rows.append({
                    'layer': layer_idx, 'head': head_idx,
                    'method': 'oracle', 'k': K_SUBSPACE,
                    'n_bits_per_dim': n_bits_sub,
                    'bits_per_vector': n_bits_sub * K_SUBSPACE,
                    'budget': f'{bits_per_dim}bit',
                    **{f'K_{m}': v for m, v in d_oracle.items()},
                })

                # --- (b) Transfer: PCA basis from domain 1, applied to domain 2 ---
                U_transfer, mean_transfer = fit_pca(k1, K_SUBSPACE)
                k2_transfer_q = subspace_polar_quantize(
                    k2_test, K_SUBSPACE, n_bits_sub, U_transfer, mean_transfer, R_sub
                )
                d_transfer = attention_score_distortion(q2_test, k2_test, k2_transfer_q)

                rows.append({
                    'layer': layer_idx, 'head': head_idx,
                    'method': 'transfer', 'k': K_SUBSPACE,
                    'n_bits_per_dim': n_bits_sub,
                    'bits_per_vector': n_bits_sub * K_SUBSPACE,
                    'budget': f'{bits_per_dim}bit',
                    **{f'K_{m}': v for m, v in d_transfer.items()},
                })

                # --- (c) Full-dim baseline ---
                R_full = random_rotation_matrix(D, seed=0)
                k2_full_q = polar_quantize(k2_test, bits_per_dim, R_full)
                d_full = attention_score_distortion(q2_test, k2_test, k2_full_q)

                rows.append({
                    'layer': layer_idx, 'head': head_idx,
                    'method': 'full_dim', 'k': D,
                    'n_bits_per_dim': bits_per_dim,
                    'bits_per_vector': bits_per_dim * D,
                    'budget': f'{bits_per_dim}bit',
                    **{f'K_{m}': v for m, v in d_full.items()},
                })

            if head_idx == 0:
                # Print 2-bit results for this layer
                oracle_kl = rows[-6]['K_kl_divergence']
                transfer_kl = rows[-5]['K_kl_divergence']
                full_kl = rows[-4]['K_kl_divergence']
                ratio = transfer_kl / oracle_kl if oracle_kl > 0 else float('inf')
                print(f"  Layer {layer_idx:2d}: oracle={oracle_kl:.6f}  "
                      f"transfer={transfer_kl:.6f}  full_dim={full_kl:.6f}  "
                      f"transfer/oracle={ratio:.2f}x")

    return rows


def print_summary(rows: list):
    print("\n" + "=" * 80)
    print("CALIBRATION TRANSFER — SUMMARY")
    print("=" * 80)

    for budget in ['2bit', '4bit']:
        budget_rows = [r for r in rows if r['budget'] == budget]
        print(f"\n--- {budget} budget ---")

        for method in ['oracle', 'transfer', 'full_dim']:
            subset = [r for r in budget_rows if r['method'] == method]
            if not subset:
                continue
            kl_vals = [r['K_kl_divergence'] for r in subset]
            top1_vals = [r['K_top1_agreement'] for r in subset]
            print(f"\n  {method}:")
            print(f"    Mean KL divergence: {np.mean(kl_vals):.6f}")
            print(f"    Median KL: {np.median(kl_vals):.6f}")
            print(f"    Mean top-1 agreement: {np.mean(top1_vals):.4f}")

            # Per layer range
            for label, lo, hi in [("Early L0-9", 0, 10), ("Mid L10-29", 10, 30), ("Late L30-39", 30, 40)]:
                sub = [r for r in subset if lo <= r['layer'] < hi]
                if sub:
                    print(f"    {label}: KL={np.mean([r['K_kl_divergence'] for r in sub]):.6f}, "
                          f"top1={np.mean([r['K_top1_agreement'] for r in sub]):.4f}")

        # Transfer penalty ratio
        oracle_map = {(r['layer'], r['head']): r for r in budget_rows if r['method'] == 'oracle'}
        transfer_map = {(r['layer'], r['head']): r for r in budget_rows if r['method'] == 'transfer'}

        ratios = []
        ratios_by_range = {'early': [], 'mid': [], 'late': []}
        for key in oracle_map:
            if key in transfer_map and oracle_map[key]['K_kl_divergence'] > 1e-10:
                ratio = transfer_map[key]['K_kl_divergence'] / oracle_map[key]['K_kl_divergence']
                ratios.append(ratio)
                layer = key[0]
                if layer < 10:
                    ratios_by_range['early'].append(ratio)
                elif layer < 30:
                    ratios_by_range['mid'].append(ratio)
                else:
                    ratios_by_range['late'].append(ratio)

        if ratios:
            print(f"\n  Transfer penalty (transfer_KL / oracle_KL):")
            print(f"    Mean ratio: {np.mean(ratios):.2f}x")
            print(f"    Median ratio: {np.median(ratios):.2f}x")
            print(f"    Min: {min(ratios):.2f}x, Max: {max(ratios):.2f}x")
            for label, key in [("Early L0-9", 'early'), ("Mid L10-29", 'mid'), ("Late L30-39", 'late')]:
                if ratios_by_range[key]:
                    print(f"    {label}: {np.mean(ratios_by_range[key]):.2f}x (median {np.median(ratios_by_range[key]):.2f}x)")

            # How often does transfer still beat full_dim?
            full_map = {(r['layer'], r['head']): r for r in budget_rows if r['method'] == 'full_dim'}
            transfer_beats_full = sum(
                1 for key in transfer_map
                if key in full_map and transfer_map[key]['K_kl_divergence'] < full_map[key]['K_kl_divergence']
            )
            print(f"\n  Transfer beats full_dim: {transfer_beats_full}/{len(transfer_map)} "
                  f"({100*transfer_beats_full/len(transfer_map):.0f}%)")


def main():
    import os
    os.chdir(Path(__file__).resolve().parent.parent)
    Path("results").mkdir(exist_ok=True)

    print("=" * 80)
    print("Calibration Transfer Experiment")
    print("Testing cross-domain PCA basis transfer for KV compression")
    print("=" * 80)

    print("\nLoading domain 1 KV vectors (fiction)...")
    kvs1 = load_kvs('results/kvs.npz')
    print(f"  {len(kvs1)} layers")

    print("Loading domain 2 KV vectors (factual)...")
    kvs2 = load_kvs('results/kvs_domain2.npz')
    print(f"  {len(kvs2)} layers")

    sample = kvs1[next(iter(kvs1.keys()))]['K']
    print(f"  Domain 1 shape: {sample.shape} (T, H, d)")
    sample2 = kvs2[next(iter(kvs2.keys()))]['K']
    print(f"  Domain 2 shape: {sample2.shape} (T, H, d)\n")

    rows = run_transfer_experiment(kvs1, kvs2)

    # Save CSV
    with open('results/calibration_transfer.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to results/calibration_transfer.csv")

    print_summary(rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
