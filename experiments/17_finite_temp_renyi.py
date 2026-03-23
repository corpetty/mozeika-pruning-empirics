"""
Experiment 17: Finite-temperature Rényi sweep over (n_replicas, T_h, rho).

Theory: n = beta_h / beta_w is the Rényi order parameter. At zero temperature
(greedy accept/reject), n does not matter — all replicas agree on the MAP
solution (confirmed in Exp 16). The sharpening effect only appears at finite
T_h where stochastic acceptance exp(-ΔE/T_h) allows exploration. Higher n
should sharpen the phase transition (lower rho_c, steeper drop).

Grid: n in [1, 2, 4, 8], T_h in [0.001, 0.003, 0.01, 0.03, 0.1],
      rho in linspace(0, 0.002, 16), 8 seeds.
Params: N=60, M=180, sigma=0.01, eta=1e-4, alpha=1.0, T=40.
"""

import numpy as np
import sys, csv, os, time

sys.path.insert(0, '/home/petty/pruning-research')

from pruning_core.data import sample_perceptron
from pruning_core.metrics import hamming_distance


# ── helpers (self-contained, matching exp 16 pattern) ─────────────────────

def energy(w, h, X, y, eta, rho, alpha=1.0):
    pred = X @ (w * h)
    L = 0.5 * np.mean((pred - y) ** 2)
    reg = 0.5 * eta * np.sum(w ** 2)
    V = alpha * np.sum(h**2 * (h - 1)**2) + 0.5 * rho * np.sum(h)
    return L + reg + V


def optimize_w_adam(w, h, X, y, eta, K=20, lr=0.01):
    """K steps of Adam on the loss w.r.t. w (h fixed)."""
    w = w.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    for k in range(1, K + 1):
        pred = X @ (w * h)
        grad = X.T @ ((pred - y) / len(y)) * h + eta * w
        m = 0.9 * m + 0.1 * grad
        v = 0.99 * v + 0.01 * grad ** 2
        m_hat = m / (1 - 0.9 ** k)
        v_hat = v / (1 - 0.99 ** k)
        w -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return w


def ensemble_energy(w_chains, h, X, y, eta, rho, alpha=1.0):
    return np.mean([energy(w, h, X, y, eta, rho, alpha) for w in w_chains])


def multi_replica_glauber_ft(X, y, h0_true, eta, rho, alpha,
                              n_replicas, T, T_h, seed):
    """
    Multi-replica Glauber with finite-temperature acceptance.
    Self-contained version with tuned Adam steps for speed.
    """
    N = X.shape[1]
    rng = np.random.default_rng(seed)

    h = np.ones(N, dtype=float)
    w_chains = [rng.normal(0, 0.1, N) for _ in range(n_replicas)]

    # Initial weight optimization
    w_chains = [optimize_w_adam(w, h, X, y, eta, K=20) for w in w_chains]

    for t in range(T):
        order = rng.permutation(N)
        for j in order:
            h_try = h.copy()
            h_try[j] = 1.0 - h_try[j]

            w_try_chains = [optimize_w_adam(w, h_try, X, y, eta, K=3)
                            for w in w_chains]

            E_curr = ensemble_energy(w_chains, h, X, y, eta, rho, alpha)
            E_try = ensemble_energy(w_try_chains, h_try, X, y, eta, rho, alpha)

            delta = E_try - E_curr
            if delta < 0:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta / T_h)

            if accept:
                h = h_try
                w_chains = w_try_chains

        # Re-optimization after each sweep
        w_chains = [optimize_w_adam(w, h, X, y, eta, K=10)
                     for w in w_chains]

    return hamming_distance(h, h0_true)


# ── parameters ────────────────────────────────────────────────────────────

N = 60
M = 180          # M/N = 3
SIGMA = 0.01     # low noise so signal dominates
ETA = 1e-4
ALPHA = 1.0
T = 20
N_SEEDS = 8

N_VALS = [1, 2, 4, 8]
T_H_VALS = [0.001, 0.003, 0.01, 0.03, 0.1]
RHO_VALS = np.linspace(0, 0.002, 16)

# ── sweep ─────────────────────────────────────────────────────────────────

t_start = time.time()
results = []
total = len(N_VALS) * len(T_H_VALS) * len(RHO_VALS) * N_SEEDS
done = 0

for n_rep in N_VALS:
    for T_h in T_H_VALS:
        print(f"\n=== n_replicas={n_rep}, T_h={T_h} ===", flush=True)
        for rho in RHO_VALS:
            hds = []
            for seed in range(N_SEEDS):
                X, y, w0, h0 = sample_perceptron(N, M, p0=0.5, sigma=SIGMA, seed=seed)
                hd = multi_replica_glauber_ft(
                    X, y, h0, ETA, rho, ALPHA, n_rep, T, T_h, seed=seed + 100
                )
                hds.append(hd)
                done += 1

            row = dict(n=n_rep, T_h=T_h, rho=rho,
                       hamming_mean=np.mean(hds),
                       hamming_std=np.std(hds))
            results.append(row)
            elapsed = time.time() - t_start
            pct = 100 * done / total
            eta_min = (elapsed / done * (total - done)) / 60 if done > 0 else 0
            print(f"  rho={rho:.5f}  hd={np.mean(hds):.3f} ± {np.std(hds):.3f}"
                  f"  [{pct:.0f}% | ETA {eta_min:.0f}m]", flush=True)

elapsed_total = time.time() - t_start
print(f"\nTotal time: {elapsed_total/60:.1f} minutes")

# ── save full results ─────────────────────────────────────────────────────

out_dir = '/home/petty/pruning-research/results'
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, 'finite_temp_renyi.csv')
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['n', 'T_h', 'rho',
                                            'hamming_mean', 'hamming_std'])
    writer.writeheader()
    writer.writerows(results)
print(f"\nSaved full results to {out_path}")

# ── extract rho_c (rho where hamming drops below 0.1) ────────────────────

rho_c_rows = []
for n_rep in N_VALS:
    for T_h in T_H_VALS:
        subset = [r for r in results if r['n'] == n_rep and r['T_h'] == T_h]
        rho_c = float('nan')
        for r in sorted(subset, key=lambda x: x['rho']):
            if r['rho'] > 0 and r['hamming_mean'] < 0.1:
                rho_c = r['rho']
                break
        rho_c_rows.append(dict(n=n_rep, T_h=T_h, rho_c=rho_c))

rho_c_path = os.path.join(out_dir, 'finite_temp_renyi_rho_c.csv')
with open(rho_c_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['n', 'T_h', 'rho_c'])
    writer.writeheader()
    writer.writerows(rho_c_rows)
print(f"Saved rho_c summary to {rho_c_path}")

# ── summary table ─────────────────────────────────────────────────────────

print("\n=== rho_c summary (rho where hamming drops below 0.1) ===\n")
for T_h in T_H_VALS:
    parts = [f"T_h={T_h:<7.3f}"]
    for n_rep in N_VALS:
        rc = next(r for r in rho_c_rows if r['n'] == n_rep and r['T_h'] == T_h)
        val = rc['rho_c']
        if np.isnan(val):
            parts.append(f"n={n_rep}: rho_c=NaN   ")
        else:
            parts.append(f"n={n_rep}: rho_c={val:.5f}")
    print("  ".join(parts))

# ── full hamming table per T_h ────────────────────────────────────────────

for T_h in T_H_VALS:
    print(f"\n--- T_h={T_h} ---")
    print(f"{'rho':>10}", end='')
    for n in N_VALS:
        print(f"  n={n:>2}", end='')
    print()
    for rho in RHO_VALS:
        print(f"{rho:>10.5f}", end='')
        for n in N_VALS:
            row = next(r for r in results
                       if r['n'] == n and r['T_h'] == T_h
                       and abs(r['rho'] - rho) < 1e-10)
            print(f"  {row['hamming_mean']:>5.3f}", end='')
        print()
