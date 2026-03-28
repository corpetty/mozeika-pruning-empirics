"""
Run compression distortion comparison using already-saved analysis.npz data.
We'll generate synthetic KV vectors that match the spectral properties we measured,
then run the subspace vs full-dim comparison.

This avoids needing to reload the 14GB model just to do numpy math.
"""

import numpy as np
import csv
from pathlib import Path
from compress import (
    polar_quantize, subspace_polar_quantize, fit_pca,
    attention_score_distortion, random_rotation_matrix
)

np.random.seed(42)


def load_analysis(path: str) -> dict:
    """Load analysis.npz and reconstruct per-layer, per-head data."""
    data = np.load(path)
    results = {}
    # Keys like L00H00_sk, L00H00_sv, L00H00_erk90, ...
    for key in data.files:
        parts = key.split('_')
        layer_head = parts[0]  # e.g. L00H00
        metric = '_'.join(parts[1:])
        layer = int(layer_head[1:3])
        head = int(layer_head[4:6])
        if layer not in results:
            results[layer] = {}
        if head not in results[layer]:
            results[layer][head] = {}
        results[layer][head][metric] = data[key]
    return results


def generate_synthetic_kvs_from_spectrum(singular_values: np.ndarray, T: int, d: int,
                                          seed: int = 0) -> np.ndarray:
    """
    Generate synthetic vectors with the given singular value spectrum.
    Returns (T, d) matrix.
    """
    rng = np.random.default_rng(seed)
    k = len(singular_values)
    # Random orthogonal U (T, k) and V (d, k)
    U = np.linalg.qr(rng.standard_normal((T, k)))[0]
    V = np.linalg.qr(rng.standard_normal((d, k)))[0]
    # Scale by singular values
    S = singular_values[:k]
    X = (U * S) @ V.T
    # Add small noise floor
    X += rng.standard_normal((T, d)) * S.min() * 0.01
    return X.astype(np.float32)


def run_compression_comparison(analysis: dict, bit_budgets: list, k_values: list,
                                T: int = 512, d_head: int = 128) -> list:
    """
    For each layer/head, generate synthetic KV vectors from measured spectrum,
    then compare full-dim vs subspace PolarQuant at matched bit budgets.
    """
    rows = []
    layers = sorted(analysis.keys())

    for layer_idx in layers:
        for head_idx in sorted(analysis[layer_idx].keys()):
            hdata = analysis[layer_idx][head_idx]
            sk = hdata.get('sk', None)
            if sk is None:
                continue

            # Generate synthetic K vectors with measured spectrum
            K = generate_synthetic_kvs_from_spectrum(sk, T, d_head, seed=layer_idx * 100 + head_idx)
            Q = generate_synthetic_kvs_from_spectrum(sk, T, d_head, seed=layer_idx * 100 + head_idx + 1)

            T_cal = T // 2
            K_cal = K[:T_cal]
            K_test = K[T_cal:]
            Q_test = Q[T_cal:]

            erk90 = float(hdata.get('erk90', d_head))
            erk95 = float(hdata.get('erk95', d_head))

            for n_bits in bit_budgets:
                # Full-dim PolarQuant
                R_full = random_rotation_matrix(d_head, seed=0)
                K_full_q = polar_quantize(K_test, n_bits, R_full)
                d_full = attention_score_distortion(Q_test, K_test, K_full_q)

                rows.append({
                    'layer': layer_idx, 'head': head_idx,
                    'method': 'full_dim', 'k': d_head, 'n_bits': n_bits,
                    'bits_per_vector': n_bits * d_head,
                    'eff_rank_K_90': erk90,
                    **{f'K_{kk}': vv for kk, vv in d_full.items()},
                })

                # Subspace PolarQuant at various k
                for k in k_values:
                    if k >= d_head:
                        continue
                    # Match total bits: n_bits_sub = round(d_head * n_bits / k)
                    n_bits_sub = max(1, round(d_head * n_bits / k))

                    U_k, mean_K = fit_pca(K_cal, k)
                    R_sub = random_rotation_matrix(k, seed=1)
                    K_sub_q = subspace_polar_quantize(K_test, k, n_bits_sub, U_k, mean_K, R_sub)
                    d_sub = attention_score_distortion(Q_test, K_test, K_sub_q)

                    rows.append({
                        'layer': layer_idx, 'head': head_idx,
                        'method': f'subspace_k{k}', 'k': k, 'n_bits': n_bits_sub,
                        'bits_per_vector': n_bits_sub * k,
                        'eff_rank_K_90': erk90,
                        **{f'K_{kk}': vv for kk, vv in d_sub.items()},
                    })

            if head_idx == 0:
                print(f"  Layer {layer_idx:2d} Head {head_idx}: eff_rank={erk90:.0f}/{d_head}  "
                      f"bits=2: full KL={rows[-len(k_values)-1-1]['K_kl_divergence']:.4f}")

    return rows


def print_summary(rows: list, bit_budgets: list, k_values: list):
    print("\n=== COMPRESSION DISTORTION SUMMARY (mean KL divergence, lower=better) ===")
    print(f"{'Method':<20}  " + "  ".join(f"{b}bit" for b in bit_budgets))
    print("-" * 60)

    methods = ['full_dim'] + [f'subspace_k{k}' for k in k_values if k < 128]
    for method in methods:
        kls = []
        for n_bits in bit_budgets:
            subset = [r for r in rows
                      if r['method'] == method and
                      (r['n_bits'] == n_bits or
                       (method == 'full_dim' and r['n_bits'] == n_bits))]
            if subset:
                kls.append(f"{np.mean([r['K_kl_divergence'] for r in subset]):.4f}")
            else:
                kls.append("  N/A ")
        print(f"{method:<20}  " + "  ".join(f"{k:>5}" for k in kls))

    # Best method per bit budget
    print("\n=== WINNER PER BIT BUDGET ===")
    for n_bits in bit_budgets:
        full = [r for r in rows if r['method'] == 'full_dim' and r['n_bits'] == n_bits]
        full_kl = np.mean([r['K_kl_divergence'] for r in full]) if full else float('inf')

        best_kl = full_kl
        best_method = 'full_dim'
        for k in k_values:
            if k >= 128:
                continue
            sub = [r for r in rows if r['method'] == f'subspace_k{k}']
            if sub:
                sub_kl = np.mean([r['K_kl_divergence'] for r in sub])
                if sub_kl < best_kl:
                    best_kl = sub_kl
                    best_method = f'subspace_k{k}'

        ratio = full_kl / best_kl if best_kl > 0 else 1.0
        print(f"  {n_bits}bit: winner={best_method} (KL={best_kl:.4f}), "
              f"vs full_dim KL={full_kl:.4f}, ratio={ratio:.2f}x")


def main():
    Path('results').mkdir(exist_ok=True)
    analysis = load_analysis('results/analysis.npz')
    print(f"Loaded analysis: {len(analysis)} layers")

    d_head = 128  # Qwen3-14B
    T = 512       # tokens per synthetic batch
    bit_budgets = [2, 4, 8]
    # k as fraction of d_head (matching eff_rank range we measured: 11-43)
    k_values = [16, 32, 48, 64]

    print(f"\nRunning compression comparison: T={T}, d_head={d_head}")
    print(f"Bit budgets: {bit_budgets}, k values: {k_values}")
    print()

    rows = run_compression_comparison(analysis, bit_budgets, k_values, T, d_head)

    # Save CSV
    with open('results/compression_distortion.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to results/compression_distortion.csv")

    print_summary(rows, bit_budgets, k_values)


if __name__ == '__main__':
    main()
