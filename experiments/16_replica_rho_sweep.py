"""
Experiment 16: Multi-replica Rényi sweep over (n_replicas, rho).

Theory: n = beta_h / beta_w is the Rényi order parameter.
  n=1  -> standard Bayesian posterior (smooth recovery above rho_c)
  n->inf -> minimax mask selection (sharpest threshold)

Implementation: n independent weight chains {w_1,...,w_n} share one mask h.
At each coordinate flip proposal for h[j]:
  1. Each chain re-optimizes its weights for the proposed mask (K Adam steps)
  2. Accept if average energy over chains decreases (zero-temperature Glauber)

This is exactly the MAP limit (beta->inf) of the Mozeika fast-learning regime,
with n replicas averaging over weight uncertainty.

Grid: n in [1, 2, 4, 8], rho in linspace(0, 0.015, 20), 8 seeds.
Params: N=60, M/N=3, eta=1e-4, alpha=1.0, T=30 sweeps.
"""

import numpy as np
import sys, csv, os
sys.path.insert(0, '/home/petty/pruning-research')

from pruning_core.data import sample_perceptron
# adam_optimize not needed — using local optimize_w_adam below
from pruning_core.metrics import hamming_distance


# ── helpers ────────────────────────────────────────────────────────────────

def energy(w, h, X, y, eta, rho, alpha=1.0):
    pred = X @ (w * h)
    L = 0.5 * np.mean((pred - y) ** 2)
    reg = 0.5 * eta * np.sum(w ** 2)
    V = alpha * np.sum(h**2 * (h - 1)**2) + 0.5 * rho * np.sum(h)
    return L + reg + V


def optimize_w_adam(w, h, X, y, eta, K=30, lr=0.01):
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


def multi_replica_glauber(X, y, h0_true, eta, rho, alpha, n_replicas, T, seed):
    """
    Multi-replica Glauber dynamics on shared mask h with n_replicas weight chains.
    Returns Hamming distance to true mask h0_true.
    """
    N = X.shape[1]
    rng = np.random.default_rng(seed)

    # Initialize: all weights active, small random weights per replica
    h = np.ones(N, dtype=float)
    w_chains = [rng.normal(0, 0.1, N) for _ in range(n_replicas)]

    # Initial weight optimization for each replica
    w_chains = [optimize_w_adam(w, h, X, y, eta, K=50) for w in w_chains]

    for t in range(T):
        order = rng.permutation(N)
        for j in order:
            # Proposed flip
            h_try = h.copy()
            h_try[j] = 1.0 - h_try[j]

            # Each replica re-optimizes weights for proposed mask (fast)
            w_try_chains = [optimize_w_adam(w, h_try, X, y, eta, K=15)
                            for w in w_chains]

            # Accept if average energy decreases (zero-T Glauber)
            E_curr = ensemble_energy(w_chains, h, X, y, eta, rho, alpha)
            E_try = ensemble_energy(w_try_chains, h_try, X, y, eta, rho, alpha)

            if E_try < E_curr:
                h = h_try
                w_chains = w_try_chains

        # Full re-optimization after each sweep
        w_chains = [optimize_w_adam(w, h, X, y, eta, K=30) for w in w_chains]

    return hamming_distance(h, h0_true)


# ── sweep ──────────────────────────────────────────────────────────────────

N = 60
M = 180        # M/N = 3
SIGMA = 0.01   # match R code — small noise so signal dominates
ETA = 1e-4
ALPHA = 1.0
T = 30
N_SEEDS = 8
N_VALS = [1, 2, 4, 8]
RHO_VALS = np.linspace(0, 0.002, 20)  # transition expected ~0.0001-0.001

results = []
total = len(N_VALS) * len(RHO_VALS) * N_SEEDS
done = 0

for n_rep in N_VALS:
    print(f"\n=== n_replicas={n_rep} ===", flush=True)
    for rho in RHO_VALS:
        hds = []
        for seed in range(N_SEEDS):
            X, y, w0, h0 = sample_perceptron(N, M, p0=0.5, sigma=SIGMA, seed=seed)
            hd = multi_replica_glauber(X, y, h0, ETA, rho, ALPHA, n_rep, T, seed=seed + 100)
            hds.append(hd)
            done += 1

        row = dict(n=n_rep, rho=rho,
                   hamming_mean=np.mean(hds),
                   hamming_std=np.std(hds),
                   hamming_min=np.min(hds),
                   hamming_max=np.max(hds))
        results.append(row)
        print(f"  rho={rho:.5f}  hd={np.mean(hds):.3f} ± {np.std(hds):.3f}", flush=True)

# ── save ───────────────────────────────────────────────────────────────────

out_path = '/home/petty/pruning-research/results/replica_rho_sweep.csv'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['n', 'rho', 'hamming_mean', 'hamming_std', 'hamming_min', 'hamming_max'])
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved to {out_path}")

# ── summary table ──────────────────────────────────────────────────────────

print(f"\n{'rho':>10}", end='')
for n in N_VALS:
    print(f"  n={n:>2}", end='')
print()
for rho in RHO_VALS:
    print(f"{rho:>10.5f}", end='')
    for n in N_VALS:
        row = next(r for r in results if r['n'] == n and abs(r['rho'] - rho) < 1e-10)
        print(f"  {row['hamming_mean']:>5.3f}", end='')
    print()
