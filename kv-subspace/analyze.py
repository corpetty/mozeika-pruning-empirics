"""
analyze.py — PCA analysis of KV vectors: effective rank, spectral decay.

Usage:
    python analyze.py --kvs results/kvs.npz --out results/analysis.npz
"""

import argparse
import numpy as np
from pathlib import Path
from collect import load_kvs


def effective_rank(singular_values: np.ndarray, threshold: float = 0.90) -> int:
    """Number of principal components needed to explain `threshold` of variance."""
    s2 = singular_values ** 2
    cumvar = np.cumsum(s2) / s2.sum()
    k = int(np.searchsorted(cumvar, threshold)) + 1
    return min(k, len(singular_values))


def analyze_layer(K: np.ndarray, V: np.ndarray):
    """
    K, V: shape (T, n_heads, d_head)
    Returns per-head analysis dict.
    """
    T, n_heads, d_head = K.shape
    results = []

    for h in range(n_heads):
        Kh = K[:, h, :]  # (T, d_head)
        Vh = V[:, h, :]

        # Center
        Kh_c = Kh - Kh.mean(axis=0)
        Vh_c = Vh - Vh.mean(axis=0)

        # SVD
        _, Sk, _ = np.linalg.svd(Kh_c, full_matrices=False)
        _, Sv, _ = np.linalg.svd(Vh_c, full_matrices=False)

        erk_90 = effective_rank(Sk, 0.90)
        erk_95 = effective_rank(Sk, 0.95)
        erv_90 = effective_rank(Sv, 0.90)
        erv_95 = effective_rank(Sv, 0.95)

        # Participation ratio (another effective rank measure)
        # PR = (sum s_i)^2 / sum(s_i^2) — bounded in [1, rank]
        pr_k = (Sk.sum() ** 2) / (Sk ** 2).sum()
        pr_v = (Sv.sum() ** 2) / (Sv ** 2).sum()

        results.append({
            'head': h,
            'eff_rank_K_90': erk_90,
            'eff_rank_K_95': erk_95,
            'eff_rank_V_90': erv_90,
            'eff_rank_V_95': erv_95,
            'participation_ratio_K': float(pr_k),
            'participation_ratio_V': float(pr_v),
            'singular_values_K': Sk,
            'singular_values_V': Sv,
            'd_head': d_head,
            'n_tokens': T,
        })

    return results


def run_analysis(kv_dict: dict) -> dict:
    """Analyze all layers. Returns nested dict layer -> head -> metrics."""
    all_results = {}
    layers = sorted(kv_dict.keys())
    print(f"Analyzing {len(layers)} layers...")

    for layer_idx in layers:
        K = kv_dict[layer_idx]['K']  # (T, n_heads, d_head)
        V = kv_dict[layer_idx]['V']
        head_results = analyze_layer(K, V)
        all_results[layer_idx] = head_results

        # Print summary
        mean_erk = np.mean([r['eff_rank_K_90'] for r in head_results])
        mean_erv = np.mean([r['eff_rank_V_90'] for r in head_results])
        d_head = head_results[0]['d_head']
        print(f"  Layer {layer_idx:2d}: mean eff_rank K={mean_erk:.1f}/{d_head}  V={mean_erv:.1f}/{d_head}  "
              f"({mean_erk/d_head*100:.0f}% / {mean_erv/d_head*100:.0f}%)")

    return all_results


def save_analysis(results: dict, path: str):
    """Save analysis results as npz."""
    flat = {}
    for layer_idx, head_results in results.items():
        for r in head_results:
            h = r['head']
            prefix = f"L{layer_idx:02d}H{h:02d}"
            flat[f"{prefix}_sk"] = r['singular_values_K']
            flat[f"{prefix}_sv"] = r['singular_values_V']
            flat[f"{prefix}_erk90"] = np.array(r['eff_rank_K_90'])
            flat[f"{prefix}_erv90"] = np.array(r['eff_rank_V_90'])
            flat[f"{prefix}_erk95"] = np.array(r['eff_rank_K_95'])
            flat[f"{prefix}_erv95"] = np.array(r['eff_rank_V_95'])
            flat[f"{prefix}_prk"] = np.array(r['participation_ratio_K'])
            flat[f"{prefix}_prv"] = np.array(r['participation_ratio_V'])
    np.savez_compressed(path, **flat)
    print(f"Saved analysis to {path}")


def print_summary(results: dict):
    """Print a compact summary table."""
    print("\n" + "="*70)
    print(f"{'Layer':>6}  {'EffRank K @90%':>14}  {'EffRank V @90%':>14}  {'d_head':>6}")
    print("-"*70)
    for layer_idx in sorted(results.keys()):
        head_results = results[layer_idx]
        d_head = head_results[0]['d_head']
        erk = np.mean([r['eff_rank_K_90'] for r in head_results])
        erv = np.mean([r['eff_rank_V_90'] for r in head_results])
        print(f"{layer_idx:>6}  {erk:>8.1f}/{d_head:<5}  {erv:>8.1f}/{d_head:<5}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kvs', default='results/kvs.npz')
    parser.add_argument('--out', default='results/analysis.npz')
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    kv_dict = load_kvs(args.kvs)
    results = run_analysis(kv_dict)
    print_summary(results)
    save_analysis(results, args.out)


if __name__ == '__main__':
    main()
