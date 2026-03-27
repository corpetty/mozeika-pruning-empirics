"""
v_compression.py — Compression distortion analysis for V vectors (complement to K analysis).

The existing compress_standalone.py only analyzes K vectors. This script runs the same
distortion analysis on V vectors using their singular value spectra (sv from analysis.npz),
and compares K vs V compression characteristics.

Usage:
    /home/petty/torch-env/bin/python3 experiments/v_compression.py
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
T = 512


def run_v_compression(analysis: dict, bit_budgets: list, k_values: list) -> list:
    """Run compression comparison for V vectors using V singular value spectra."""
    rows = []
    layers = sorted(analysis.keys())

    for layer_idx in layers:
        for head_idx in sorted(analysis[layer_idx].keys()):
            hdata = analysis[layer_idx][head_idx]
            sv = hdata.get('sv', None)
            sk = hdata.get('sk', None)
            if sv is None:
                continue

            erv90 = float(hdata.get('erv90', D_HEAD))
            erk90 = float(hdata.get('erk90', D_HEAD))

            # Generate synthetic V vectors with V spectrum
            # For V distortion, we use the value-weighted attention score:
            # The distortion metric is still attention_score_distortion,
            # but applied as: we use Q from K spectrum (queries), K from K spectrum (keys),
            # but then compress V separately. However the standard V distortion metric is:
            # || softmax(QK^T/sqrt(d)) * V - softmax(QK^T/sqrt(d)) * V_compressed ||
            # For simplicity and comparability, we use the same dot-product distortion:
            # generate synthetic "Q" vectors (as the attention-weighted sum query),
            # and measure distortion as attention_score_distortion(Q_v, V_true, V_compressed)
            V = generate_synthetic_kvs_from_spectrum(sv, T, D_HEAD, seed=layer_idx * 100 + head_idx + 50)
            Q_v = generate_synthetic_kvs_from_spectrum(sv, T, D_HEAD, seed=layer_idx * 100 + head_idx + 51)

            T_cal = T // 2
            V_cal = V[:T_cal]
            V_test = V[T_cal:]
            Q_test = Q_v[T_cal:]

            for n_bits in bit_budgets:
                # Full-dim PolarQuant on V
                R_full = random_rotation_matrix(D_HEAD, seed=0)
                V_full_q = polar_quantize(V_test, n_bits, R_full)
                d_full = attention_score_distortion(Q_test, V_test, V_full_q)

                rows.append({
                    'layer': layer_idx, 'head': head_idx,
                    'kv_type': 'V',
                    'method': 'full_dim', 'k': D_HEAD, 'n_bits': n_bits,
                    'bits_per_vector': n_bits * D_HEAD,
                    'eff_rank_V_90': erv90,
                    'eff_rank_K_90': erk90,
                    **{f'V_{m}': v for m, v in d_full.items()},
                })

                # Subspace PolarQuant at various k
                for k in k_values:
                    if k >= D_HEAD:
                        continue
                    n_bits_sub = max(1, round(D_HEAD * n_bits / k))

                    U_k, mean_V = fit_pca(V_cal, k)
                    R_sub = random_rotation_matrix(k, seed=1)
                    V_sub_q = subspace_polar_quantize(V_test, k, n_bits_sub, U_k, mean_V, R_sub)
                    d_sub = attention_score_distortion(Q_test, V_test, V_sub_q)

                    rows.append({
                        'layer': layer_idx, 'head': head_idx,
                        'kv_type': 'V',
                        'method': f'subspace_k{k}', 'k': k, 'n_bits': n_bits_sub,
                        'bits_per_vector': n_bits_sub * k,
                        'eff_rank_V_90': erv90,
                        'eff_rank_K_90': erk90,
                        **{f'V_{m}': v for m, v in d_sub.items()},
                    })

            if head_idx == 0:
                # Print the 2-bit row for this head
                full_row = [r for r in rows if r['layer'] == layer_idx and r['head'] == head_idx
                            and r['method'] == 'full_dim' and r['n_bits'] == 2]
                k64_row = [r for r in rows if r['layer'] == layer_idx and r['head'] == head_idx
                           and r['method'] == 'subspace_k64' and r['n_bits'] == 4]
                if full_row and k64_row:
                    print(f"  Layer {layer_idx:2d}: V eff_rank_90={erv90:4.0f}  "
                          f"2bit: full_KL={full_row[0]['V_kl_divergence']:.6f}  "
                          f"k64_KL={k64_row[0]['V_kl_divergence']:.6f}")

    return rows


def load_k_results() -> list:
    """Load existing K compression results for comparison."""
    k_rows = []
    csv_path = 'results/compression_distortion.csv'
    if not Path(csv_path).exists():
        return k_rows
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            k_rows.append(row)
    return k_rows


def print_summary(v_rows: list, k_rows: list, bit_budgets: list, k_values: list):
    print("\n" + "=" * 80)
    print("V-VECTOR COMPRESSION — SUMMARY (mean KL divergence, lower=better)")
    print("=" * 80)

    # V summary table
    methods = ['full_dim'] + [f'subspace_k{k}' for k in k_values if k < 128]
    print(f"\n{'Method':<20}  " + "  ".join(f"{b}bit" for b in bit_budgets))
    print("-" * 60)

    for method in methods:
        kls = []
        for n_bits in bit_budgets:
            subset = [r for r in v_rows if r['method'] == method and r['n_bits'] == n_bits]
            if not subset:
                # For subspace methods, n_bits is adjusted; match by full_dim bit budget
                if 'subspace' in method:
                    k_val = int(method.split('k')[1])
                    n_bits_sub = max(1, round(D_HEAD * n_bits / k_val))
                    subset = [r for r in v_rows if r['method'] == method and r['n_bits'] == n_bits_sub]
            if subset:
                kls.append(f"{np.mean([r['V_kl_divergence'] for r in subset]):.4f}")
            else:
                kls.append("  N/A ")
        print(f"{method:<20}  " + "  ".join(f"{k:>6}" for k in kls))

    # K vs V comparison
    if k_rows:
        print("\n" + "=" * 80)
        print("K vs V COMPARISON (at 2-bit budget, mean KL)")
        print("=" * 80)

        for method in methods:
            v_sub = [r for r in v_rows if r['method'] == method]
            if 'subspace' in method:
                k_val = int(method.split('k')[1])
                n_bits_sub = max(1, round(D_HEAD * 2 / k_val))
                v_sub = [r for r in v_sub if r['n_bits'] == n_bits_sub]
            else:
                v_sub = [r for r in v_sub if r['n_bits'] == 2]

            k_sub = [r for r in k_rows if r['method'] == method]
            if 'subspace' in method:
                k_sub = [r for r in k_sub if int(r['n_bits']) == n_bits_sub]
            else:
                k_sub = [r for r in k_sub if int(r['n_bits']) == 2]

            if v_sub and k_sub:
                v_kl = np.mean([r['V_kl_divergence'] for r in v_sub])
                k_kl = np.mean([float(r['K_kl_divergence']) for r in k_sub])
                ratio = v_kl / k_kl if k_kl > 0 else float('inf')
                winner = "V better" if v_kl < k_kl else "K better"
                print(f"  {method:<20}: K_KL={k_kl:.4f}, V_KL={v_kl:.4f}, ratio={ratio:.2f}x ({winner})")

    # Per-layer V stats
    print("\n--- V compression per layer range (2bit budget, k=64 subspace) ---")
    for label, lo, hi in [("Early L0-9", 0, 10), ("Mid L10-29", 10, 30), ("Late L30-39", 30, 40)]:
        v_sub = [r for r in v_rows if r['method'] == 'subspace_k64'
                 and lo <= r['layer'] < hi
                 and r['n_bits'] == max(1, round(D_HEAD * 2 / 64))]
        if v_sub:
            kl = np.mean([r['V_kl_divergence'] for r in v_sub])
            top1 = np.mean([r['V_top1_agreement'] for r in v_sub])
            erv = np.mean([r['eff_rank_V_90'] for r in v_sub])
            print(f"  {label}: V_KL={kl:.4f}, top1={top1:.4f}, mean_eff_rank_V={erv:.1f}")

    # V eff_rank distribution
    erv_vals = [r['eff_rank_V_90'] for r in v_rows if r['method'] == 'full_dim' and r['n_bits'] == 2]
    erk_vals = [r['eff_rank_K_90'] for r in v_rows if r['method'] == 'full_dim' and r['n_bits'] == 2]
    if erv_vals:
        print(f"\n  V eff_rank_90: min={min(erv_vals):.0f}, max={max(erv_vals):.0f}, mean={np.mean(erv_vals):.1f}")
        print(f"  K eff_rank_90: min={min(erk_vals):.0f}, max={max(erk_vals):.0f}, mean={np.mean(erk_vals):.1f}")


def main():
    import os
    os.chdir(Path(__file__).resolve().parent.parent)
    Path("results").mkdir(exist_ok=True)

    print("=" * 80)
    print("V-Vector Compression Experiment")
    print("Comparing full-dim vs subspace PolarQuant for V vectors")
    print("=" * 80)

    analysis = load_analysis('results/analysis.npz')
    print(f"Loaded analysis: {len(analysis)} layers\n")

    bit_budgets = [2, 4, 8]
    k_values = [16, 32, 48, 64]

    v_rows = run_v_compression(analysis, bit_budgets, k_values)

    # Save CSV
    with open('results/v_compression_distortion.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=v_rows[0].keys())
        writer.writeheader()
        writer.writerows(v_rows)
    print(f"\nSaved {len(v_rows)} rows to results/v_compression_distortion.csv")

    # Load K results for comparison
    k_rows = load_k_results()
    print(f"Loaded {len(k_rows)} K compression rows for comparison")

    print_summary(v_rows, k_rows, bit_budgets, k_values)
    print("\nDone.")


if __name__ == "__main__":
    main()
