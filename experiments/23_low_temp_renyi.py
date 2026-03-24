"""
Experiment 23: Low-temperature Rényi window.

Prior result (Exp 17): T_h=0.001–0.1 is too hot. Energy differences from mask
flips are O(sigma²/N) ≈ 10^-5 for sigma=0.01, N=60. Need T_h ≪ 10^-3.

Goal: Find the narrow temperature window where finite-T_h stochastic acceptance
improves over MAP (greedy). If it exists, characterize Rényi sharpening there.

Grid: T_h in [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
      n in [1, 2, 4, 8],
      rho in linspace(0, 0.0005, 12),
      6 seeds.
Params: N=60, M=180, sigma=0.01, eta=1e-4, alpha=1.0, T=25.

Optimization: For T_h=0 (MAP baseline), only run n=1 and duplicate results
for n=2,4,8 — we know from Exp 16 that replicas don't matter at zero temp.
"""

import numpy as np
import sys, csv, os, time

sys.path.insert(0, '/home/petty/pruning-research')

from pruning_core.data import sample_perceptron
from pruning_core.metrics import hamming_distance


# ── precomputed helpers (O(N^2) inner loop) ──────────────────────────────

def precompute(X, y):
    M = X.shape[0]
    A = X.T @ X / M
    b = X.T @ y / M
    c = np.dot(y, y) / M
    return A, b, c


def energy_fast(w, h, A, b, c, eta, rho, alpha=1.0):
    wh = w * h
    L = 0.5 * (wh @ A @ wh - 2.0 * b @ wh + c)
    reg = 0.5 * eta * np.sum(w ** 2)
    V = alpha * np.sum(h**2 * (h - 1)**2) + 0.5 * rho * np.sum(h)
    return L + reg + V


def optimize_w_fast(w, h, A, b, eta, K=30, lr=0.01):
    w = w.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    for k in range(1, K + 1):
        wh = w * h
        grad = (A @ wh) * h - b * h + eta * w
        m = 0.9 * m + 0.1 * grad
        v = 0.99 * v + 0.01 * grad ** 2
        m_hat = m / (1 - 0.9 ** k)
        v_hat = v / (1 - 0.99 ** k)
        w -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return w


def ensemble_energy_fast(w_chains, h, A, b, c, eta, rho, alpha=1.0):
    return np.mean([energy_fast(w, h, A, b, c, eta, rho, alpha)
                     for w in w_chains])


# ── core: multi-replica Glauber with finite-T and precomputed stats ──────

def run_finite_temp_replica(A, b, c, h0_true, N, eta, rho, alpha,
                             n_replicas, T, T_h, seed):
    """
    Multi-replica Glauber with Metropolis acceptance at temperature T_h.
    Uses precomputed sufficient statistics for O(N^2) inner loop.

    T_h=0 means greedy (MAP). T_h>0 means stochastic acceptance.
    """
    rng = np.random.default_rng(seed)

    h = np.ones(N, dtype=float)
    w_chains = [rng.normal(0, 0.1, N) for _ in range(n_replicas)]

    # Initial weight optimization
    w_chains = [optimize_w_fast(w, h, A, b, eta, K=50) for w in w_chains]

    for t in range(T):
        order = rng.permutation(N)
        n_flips = 0
        for j in order:
            # Compute current energy before flip
            E_curr = ensemble_energy_fast(w_chains, h, A, b, c, eta, rho, alpha)

            # Flip
            old_hj = h[j]
            h[j] = 1.0 - old_hj

            w_try_chains = [optimize_w_fast(w, h, A, b, eta, K=5)
                            for w in w_chains]

            E_try = ensemble_energy_fast(w_try_chains, h, A, b, c, eta, rho, alpha)
            delta = E_try - E_curr

            if T_h == 0:
                accept = delta < 0
            elif delta < 0:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta / T_h)

            if accept:
                w_chains = w_try_chains
                n_flips += 1
            else:
                h[j] = old_hj  # revert

        # Re-optimization after each sweep
        w_chains = [optimize_w_fast(w, h, A, b, eta, K=15)
                     for w in w_chains]

        # Early stopping for MAP: no flips = converged
        if T_h == 0 and n_flips == 0 and t > 3:
            break

    return hamming_distance(h, h0_true)


# ── parameters ────────────────────────────────────────────────────────────

N = 60
M = 180
SIGMA = 0.01
ETA = 1e-4
ALPHA = 1.0
T = 25
N_SEEDS = 6

T_H_VALS = [0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]  # 0 = MAP baseline
N_VALS = [1, 2, 4, 8]
RHO_VALS = np.linspace(0, 0.0005, 12)

# ── precompute datasets ──────────────────────────────────────────────────

print("Experiment 23: Low-temperature Rényi window")
print(f"N={N}, M={M}, sigma={SIGMA}, eta={ETA}, alpha={ALPHA}, T={T}")
print(f"T_h values: {T_H_VALS}")
print(f"n_replicas: {N_VALS}")
print(f"rho values: {[f'{r:.5f}' for r in RHO_VALS]}")
print(f"Seeds: {N_SEEDS}")
print(f"Optimization: MAP baseline (T_h=0) only runs n=1")
print()

datasets = []
for seed in range(N_SEEDS):
    X, y, w0, h0 = sample_perceptron(N, M, p0=0.5, sigma=SIGMA, seed=seed)
    A, b, c = precompute(X, y)
    datasets.append((A, b, c, h0))

# ── sweep ─────────────────────────────────────────────────────────────────

t_start = time.time()
results = []

# Count total jobs (T_h=0: only n=1; others: all n)
total_jobs = len(RHO_VALS) * N_SEEDS  # T_h=0, n=1
for T_h in T_H_VALS[1:]:  # finite T_h
    total_jobs += len(N_VALS) * len(RHO_VALS) * N_SEEDS
done = 0

for T_h in T_H_VALS:
    if T_h == 0:
        # MAP baseline: only run n=1, duplicate for others
        n_rep = 1
        T_h_label = "MAP"
        print(f"\n=== T_h={T_h_label}, n_replicas=1 (skip n>1 — known identical at T=0) ===",
              flush=True)
        map_results_by_rho = {}
        for rho in RHO_VALS:
            hds = []
            for seed in range(N_SEEDS):
                A, b, c, h0 = datasets[seed]
                hd = run_finite_temp_replica(
                    A, b, c, h0, N, ETA, rho, ALPHA,
                    n_rep, T, T_h, seed=seed + 200
                )
                hds.append(hd)
                done += 1

            hm, hs = np.mean(hds), np.std(hds)
            map_results_by_rho[rho] = (hm, hs)

            # Store for all n values
            for n in N_VALS:
                results.append(dict(T_h=T_h, n=n, rho=rho,
                                     hamming_mean=hm, hamming_std=hs))

            elapsed = time.time() - t_start
            pct = 100 * done / total_jobs
            eta_min = (elapsed / done * (total_jobs - done)) / 60 if done > 0 else 0
            print(f"  rho={rho:.5f}  hd={hm:.4f} ± {hs:.4f}"
                  f"  [{pct:.0f}% | ETA {eta_min:.0f}m]", flush=True)
    else:
        # Finite temperature: run all n values
        for n_rep in N_VALS:
            T_h_label = f"{T_h:.0e}"
            print(f"\n=== T_h={T_h_label}, n_replicas={n_rep} ===", flush=True)
            for rho in RHO_VALS:
                hds = []
                for seed in range(N_SEEDS):
                    A, b, c, h0 = datasets[seed]
                    hd = run_finite_temp_replica(
                        A, b, c, h0, N, ETA, rho, ALPHA,
                        n_rep, T, T_h, seed=seed + 200
                    )
                    hds.append(hd)
                    done += 1

                row = dict(T_h=T_h, n=n_rep, rho=rho,
                           hamming_mean=np.mean(hds),
                           hamming_std=np.std(hds))
                results.append(row)
                elapsed = time.time() - t_start
                pct = 100 * done / total_jobs
                eta_min = (elapsed / done * (total_jobs - done)) / 60 if done > 0 else 0
                print(f"  rho={rho:.5f}  hd={np.mean(hds):.4f} ± {np.std(hds):.4f}"
                      f"  [{pct:.0f}% | ETA {eta_min:.0f}m]", flush=True)

elapsed_total = time.time() - t_start
print(f"\nTotal time: {elapsed_total/60:.1f} minutes")

# ── save full results ─────────────────────────────────────────────────────

out_dir = '/home/petty/pruning-research/results'
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, 'low_temp_renyi.csv')
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['T_h', 'n', 'rho',
                                            'hamming_mean', 'hamming_std'])
    writer.writeheader()
    writer.writerows(results)
print(f"\nSaved to {out_path}")

# ── summary table: Hamming at rho_c (0.0001) for each (T_h, n) ──────────

# Find the rho closest to 0.0001
rho_c_target = 0.0001
rho_nearest = min(RHO_VALS, key=lambda r: abs(r - rho_c_target))

print(f"\n{'='*70}")
print(f"Hamming at rho ≈ {rho_nearest:.5f} (nearest to rho_c=0.0001)")
print(f"{'='*70}")
print(f"{'T_h':>10}", end='')
for n in N_VALS:
    print(f"   n={n:>2}", end='')
print()
print("-" * 50)

for T_h in T_H_VALS:
    T_h_label = "MAP" if T_h == 0 else f"{T_h:.0e}"
    print(f"{T_h_label:>10}", end='')
    for n in N_VALS:
        row = next((r for r in results
                     if r['T_h'] == T_h and r['n'] == n
                     and abs(r['rho'] - rho_nearest) < 1e-10), None)
        if row:
            print(f"  {row['hamming_mean']:.4f}", end='')
        else:
            print(f"     NaN", end='')
    print()

# ── full hamming table per T_h ────────────────────────────────────────────

for T_h in T_H_VALS:
    T_h_label = "MAP" if T_h == 0 else f"{T_h:.0e}"
    print(f"\n--- T_h={T_h_label} ---")
    print(f"{'rho':>10}", end='')
    for n in N_VALS:
        print(f"  n={n:>2}", end='')
    print()
    for rho in RHO_VALS:
        print(f"{rho:>10.5f}", end='')
        for n in N_VALS:
            row = next((r for r in results
                         if r['T_h'] == T_h and r['n'] == n
                         and abs(r['rho'] - rho) < 1e-10), None)
            if row:
                print(f"  {row['hamming_mean']:>5.4f}", end='')
            else:
                print(f"    NaN", end='')
        print()

# ── identify sweet spot ───────────────────────────────────────────────────

print(f"\n{'='*70}")
print("ANALYSIS: Best T_h per n (lowest Hamming at rho_c)")
print(f"{'='*70}")

for n in N_VALS:
    best_T_h = None
    best_hd = 1.0
    for T_h in T_H_VALS:
        row = next((r for r in results
                     if r['T_h'] == T_h and r['n'] == n
                     and abs(r['rho'] - rho_nearest) < 1e-10), None)
        if row and row['hamming_mean'] < best_hd:
            best_hd = row['hamming_mean']
            best_T_h = T_h
    T_h_label = "MAP" if best_T_h == 0 else f"{best_T_h:.0e}"
    print(f"  n={n}: best T_h={T_h_label}, Hamming={best_hd:.4f}")

# ── does finite T_h ever beat MAP? ──────────────────────────────────────

print(f"\n{'='*70}")
print("KEY QUESTION: Does any finite T_h beat MAP?")
print(f"{'='*70}")

map_hd = next(r for r in results if r['T_h'] == 0 and r['n'] == 1
              and abs(r['rho'] - rho_nearest) < 1e-10)['hamming_mean']
print(f"MAP baseline Hamming at rho≈{rho_nearest:.5f}: {map_hd:.4f}")

any_improvement = False
for T_h in T_H_VALS[1:]:
    for n in N_VALS:
        row = next((r for r in results
                     if r['T_h'] == T_h and r['n'] == n
                     and abs(r['rho'] - rho_nearest) < 1e-10), None)
        if row and row['hamming_mean'] < map_hd - 0.005:
            print(f"  IMPROVEMENT: T_h={T_h:.0e}, n={n} -> Hamming={row['hamming_mean']:.4f} "
                  f"(delta={map_hd - row['hamming_mean']:.4f})")
            any_improvement = True

if not any_improvement:
    print("  No finite T_h improves over MAP at this rho. The Rényi window may not exist")
    print("  at these parameters, or it requires even finer temperature resolution.")
