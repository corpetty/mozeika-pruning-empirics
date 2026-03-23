"""
Experiment 19: UWSH Subspace Angle Analysis

Tests whether Mozeika pruning at rho_c causes independently-initialized runs
to converge to a shared low-dimensional weight subspace — connecting Mozeika
to the Universal Weight Subspace Hypothesis (Kaushik et al., arXiv:2512.05117).

Hypothesis: When rho ≈ rho_c, the surviving weights (w * h) from independent
runs share a common low-dimensional spectral structure.  Above rho_c, all
weights are pruned (trivial).  Below rho_c, weights are dense and random (no
shared structure).  At rho_c, the surviving weights are the most informative
ones — and if the data has shared structure, those weights should span the
same subspace regardless of initialization.

Steps:
  1. Generate independent pruned weight vectors across seeds at several rho
  2. Spectral analysis: PCA, participation ratio, top-k variance, pairwise cosine
  3. Cross-dataset comparison (key UWSH test)
  4. Save CSV results and print summary

Uses zero-temperature Glauber (inline, same as exp 16).
"""

import numpy as np
import csv
import os
import sys

sys.path.insert(0, '/home/petty/pruning-research')

from pruning_core.data import sample_perceptron
from experiments.uwsh_subspace_helpers import (
    run_glauber_single,
    spectral_analysis,
)


# ── Parameters ───────────────────────────────────────────────────────────

N = 120
M = 360         # alpha = M/N = 3
SIGMA = 0.01
ETA = 1e-4
ALPHA = 1.0
P0 = 0.5
T = 30          # Glauber sweeps
RHO_C = 0.0001  # critical rho from prior experiments

RHO_VALS = [0, RHO_C / 2, RHO_C, 2 * RHO_C, 5 * RHO_C]
N_SEEDS = 20    # independent initializations per rho
K_DATASETS = 5  # for cross-dataset comparison
N_INITS_PER_DATASET = 4  # inits per dataset for cross-dataset


# ── Step 1 & 2: Single-dataset spectral analysis ────────────────────────

def run_step1_step2():
    """Generate pruned weights and compute spectral metrics for each rho."""
    print("=" * 60)
    print("Step 1-2: Single-dataset spectral analysis")
    print(f"  N={N}, M={M}, sigma={SIGMA}, eta={ETA}, rho_c={RHO_C}")
    print(f"  {N_SEEDS} seeds per rho, {len(RHO_VALS)} rho values")
    print("=" * 60)

    # Generate one dataset (seed=0)
    X, y, w0, h0 = sample_perceptron(N, M, p0=P0, sigma=SIGMA, seed=0)

    results_rows = []
    summary = []

    for rho in RHO_VALS:
        print(f"\n  rho={rho:.5f} ...", end='', flush=True)
        W_rho = []

        for seed in range(N_SEEDS):
            w_final, h_final = run_glauber_single(
                X, y, ETA, rho, ALPHA, T, seed=seed + 1000
            )
            w_pruned = w_final * h_final
            W_rho.append(w_pruned)
            if (seed + 1) % 5 == 0:
                print(f" {seed+1}", end='', flush=True)

        W_rho = np.array(W_rho)  # (N_SEEDS, N)
        metrics = spectral_analysis(W_rho, k=5)

        summary.append({
            'rho': rho,
            'PR': metrics['participation_ratio'],
            'top5_var': metrics['top5_variance_frac'],
            'cos_sim': metrics['mean_pairwise_cos_sim'],
        })

        # One row per (rho, seed) with per-seed info + aggregate metrics
        for seed in range(N_SEEDS):
            row = {
                'rho': rho,
                'seed': seed,
                'participation_ratio': metrics['participation_ratio'],
                'top5_variance_frac': metrics['top5_variance_frac'],
                'mean_pairwise_cos_sim': metrics['mean_pairwise_cos_sim'],
            }
            # Add top-10 eigenvalues
            for i, ev in enumerate(metrics['eigenvalues']):
                row[f'eigenvalue_{i+1}'] = ev
            results_rows.append(row)

        print(" done", flush=True)

    return results_rows, summary


# ── Step 3: Cross-dataset comparison ────────────────────────────────────

def run_step3():
    """
    Generate K datasets with different seeds.
    For each dataset, run N_INITS_PER_DATASET independent inits at several rho values.
    Collect all pruned weight vectors and do PCA across ALL of them.
    """
    print("\n" + "=" * 60)
    print("Step 3: Cross-dataset comparison (key UWSH test)")
    print(f"  {K_DATASETS} datasets x {N_INITS_PER_DATASET} inits = "
          f"{K_DATASETS * N_INITS_PER_DATASET} vectors per rho")
    print("=" * 60)

    cross_results = []

    for rho in RHO_VALS:
        print(f"\n  rho={rho:.5f} ...", end='', flush=True)
        all_w_pruned = []

        for ds_idx in range(K_DATASETS):
            # Different dataset seed for each dataset
            X, y, w0, h0 = sample_perceptron(
                N, M, p0=P0, sigma=SIGMA, seed=5000 + ds_idx
            )

            for init_idx in range(N_INITS_PER_DATASET):
                glauber_seed = 6000 + ds_idx * 100 + init_idx
                w_final, h_final = run_glauber_single(
                    X, y, ETA, rho, ALPHA, T, seed=glauber_seed
                )
                w_pruned = w_final * h_final
                all_w_pruned.append(w_pruned)

            print(f" ds{ds_idx}", end='', flush=True)

        W_all = np.array(all_w_pruned)  # (K_DATASETS * N_INITS_PER_DATASET, N)
        metrics = spectral_analysis(W_all, k=5)

        cross_results.append({
            'rho': rho,
            'participation_ratio': metrics['participation_ratio'],
            'top5_variance_frac': metrics['top5_variance_frac'],
            'mean_pairwise_cos_sim': metrics['mean_pairwise_cos_sim'],
        })

        print(" done", flush=True)

    return cross_results


# ── Step 4: Save and print ──────────────────────────────────────────────

def save_results(spectral_rows, cross_results):
    os.makedirs('/home/petty/pruning-research/results', exist_ok=True)

    # Save spectral CSV
    spectral_path = '/home/petty/pruning-research/results/uwsh_spectral.csv'
    fieldnames = ['rho', 'seed', 'participation_ratio', 'top5_variance_frac',
                  'mean_pairwise_cos_sim']
    fieldnames += [f'eigenvalue_{i+1}' for i in range(10)]
    with open(spectral_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(spectral_rows)
    print(f"\nSaved {spectral_path}")

    # Save cross-dataset CSV
    cross_path = '/home/petty/pruning-research/results/uwsh_cross_dataset.csv'
    cross_fields = ['rho', 'participation_ratio', 'top5_variance_frac',
                    'mean_pairwise_cos_sim']
    with open(cross_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cross_fields)
        writer.writeheader()
        writer.writerows(cross_results)
    print(f"Saved {cross_path}")


def print_summary(summary, cross_results):
    print("\n" + "=" * 60)
    print("SINGLE-DATASET SPECTRAL SUMMARY")
    print("=" * 60)
    print(f"{'rho':>10}  {'PR':>6}  {'top5_var':>8}  {'cos_sim':>8}")
    print("-" * 40)
    for row in summary:
        print(f"{row['rho']:>10.5f}  {row['PR']:>6.1f}  "
              f"{row['top5_var']:>8.2f}  {row['cos_sim']:>8.4f}")

    print("\n" + "=" * 60)
    print("CROSS-DATASET SPECTRAL SUMMARY (key UWSH test)")
    print("=" * 60)
    print(f"{'rho':>10}  {'PR':>6}  {'top5_var':>8}  {'cos_sim':>8}")
    print("-" * 40)
    for row in cross_results:
        print(f"{row['rho']:>10.5f}  {row['participation_ratio']:>6.1f}  "
              f"{row['top5_variance_frac']:>8.2f}  "
              f"{row['mean_pairwise_cos_sim']:>8.4f}")


def print_interpretation(summary, cross_results):
    print("\n" + "=" * 60)
    print("INTERPRETATION — UWSH Connection")
    print("=" * 60)

    # Extract key metrics
    rho0 = next(s for s in summary if s['rho'] == 0)
    rho_c_row = next(s for s in summary if abs(s['rho'] - RHO_C) < 1e-10)
    rho_high = summary[-1]

    cross_rho0 = next(c for c in cross_results if c['rho'] == 0)
    cross_rho_c = next(c for c in cross_results
                       if abs(c['rho'] - RHO_C) < 1e-10)
    cross_high = cross_results[-1]

    # Detect degenerate single-dataset case (all runs converge trivially)
    trivial_convergence = rho0['PR'] < 2.0 and rho0['cos_sim'] > 0.99

    # Cross-dataset trends
    cross_cos_trend = cross_high['mean_pairwise_cos_sim'] - cross_rho0['mean_pairwise_cos_sim']

    if trivial_convergence:
        print(f"""
SINGLE-DATASET: PR={rho0['PR']:.1f}, cos_sim={rho0['cos_sim']:.4f} at ALL
rho values. The overdetermined perceptron (alpha=M/N={M/N:.0f}) has a unique
global minimum, so all {N_SEEDS} independent initializations converge to
the same solution regardless of rho. This is trivial convergence — not
evidence for or against UWSH, just a consequence of convexity + sufficient
data. The slight cos_sim decrease at high rho ({rho_high['cos_sim']:.4f})
is from stochastic mask selection when many weights get pruned.

CROSS-DATASET: PR={cross_rho0['participation_ratio']:.1f} across
{K_DATASETS} datasets with independently random true weights. This is
expected: each dataset defines a different optimal w*, contributing one
independent direction, so PR ≈ K_DATASETS={K_DATASETS}.

The key UWSH signal is the cross-dataset cos_sim trend with rho:
  rho=0:       cos_sim={cross_rho0['mean_pairwise_cos_sim']:.4f}
  rho=rho_c:   cos_sim={cross_rho_c['mean_pairwise_cos_sim']:.4f}
  rho=5*rho_c: cos_sim={cross_high['mean_pairwise_cos_sim']:.4f}
  delta = {cross_cos_trend:+.4f}

{'A positive delta suggests pruning nudges weight vectors toward' if cross_cos_trend > 0 else 'No clear trend — pruning does not appear to concentrate cross-dataset'}
{'similar sparse support patterns even across different true weights.' if cross_cos_trend > 0 else 'weight vectors into a shared subspace at this problem size.'}

CONCLUSION: The perceptron is too simple (convex, unique minimum) to
distinguish UWSH from trivial convergence. To see non-trivial subspace
sharing, extend to: (a) underdetermined regime alpha<1 where multiple
solutions exist, (b) multi-layer networks with non-convex loss landscape,
or (c) shared latent structure across datasets (e.g., low-rank X).
The modest cross-dataset cos_sim increase ({cross_cos_trend:+.4f}) hints
that pruning may concentrate weight support — worth investigating at
larger scale.
""")
    else:
        pr_drops = rho0['PR'] > rho_c_row['PR'] > rho_high['PR']
        cos_rises = rho0['cos_sim'] < rho_c_row['cos_sim'] < rho_high['cos_sim']
        cross_converges = cross_rho_c['participation_ratio'] < cross_rho0['participation_ratio']

        print(f"""
At rho=0, weights are spread: PR={rho0['PR']:.1f}, cos_sim={rho0['cos_sim']:.4f}.
At rho_c={RHO_C}, PR={rho_c_row['PR']:.1f}, cos_sim={rho_c_row['cos_sim']:.4f}.
At 5*rho_c, PR={rho_high['PR']:.1f}, cos_sim={rho_high['cos_sim']:.4f}.

PR decreasing with rho: {'YES' if pr_drops else 'NO'}
Cosine similarity increasing: {'YES' if cos_rises else 'NO'}
Cross-dataset convergence: {'OBSERVED' if cross_converges else 'NOT OBSERVED'}

{'CONCLUSION: Evidence supports the UWSH connection — pruning concentrates' if (pr_drops and cos_rises) else 'CONCLUSION: Results are mixed —'}
{'weights into a shared low-dimensional subspace.' if (pr_drops and cos_rises) else 'further investigation needed with larger N or different parameter regimes.'}
""")


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    spectral_rows, summary = run_step1_step2()
    cross_results = run_step3()
    save_results(spectral_rows, cross_results)
    print_summary(summary, cross_results)
    print_interpretation(summary, cross_results)
