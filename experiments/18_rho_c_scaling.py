"""
Experiment 18: rho_c scaling law across (N, M/N, sigma, eta).

Goal: Pin down the empirical rho_c scaling law and compare to
the Mozeika theoretical prediction rho_c ≈ 2*sqrt(alpha*eta).

Method:
  - For each (N, alpha_ratio, sigma, eta), sweep a fixed geometric rho grid.
  - At each rho, run zero-temp Glauber over N_SEEDS seeds, measure mean hamming.
  - Extract rho_c = first rho where mean hamming < 0.15 (with log-interpolation).
  - Early-exit once transition found.
  - Fit power law: rho_c ~ C * N^a * (M/N)^b * sigma^c * eta^d

Key optimization: precompute A = X^T X / M and b = X^T y / M so that
energy and gradient computations are O(N^2) instead of O(NM). This
gives ~alpha_ratio speedup for the inner Adam loop.

Grid: N in [30,60,120,240], alpha_ratio in [1,2,4,8],
      sigma in [0.001,0.01,0.1], eta in [1e-5,1e-4,1e-3]
      → 144 combos × 4 seeds
"""

import numpy as np
import sys, csv, os, time

sys.path.insert(0, '/home/petty/pruning-research')

from pruning_core.data import sample_perceptron
from pruning_core.metrics import hamming_distance


# ── precomputed helpers (O(N^2) instead of O(NM)) ────────────────────────

def precompute(X, y):
    """Precompute sufficient statistics for O(N^2) energy/gradient."""
    M = X.shape[0]
    A = X.T @ X / M          # (N, N) — symmetric
    b = X.T @ y / M          # (N,)
    c = np.dot(y, y) / M     # scalar
    return A, b, c


def energy_fast(w, h, A, b, c, eta, rho, alpha=1.0):
    wh = w * h
    L = 0.5 * (wh @ A @ wh - 2.0 * b @ wh + c)
    reg = 0.5 * eta * np.sum(w ** 2)
    V = alpha * np.sum(h**2 * (h - 1)**2) + 0.5 * rho * np.sum(h)
    return L + reg + V


def optimize_w_fast(w, h, A, b, eta, K=30, lr=0.01):
    """K steps of Adam using precomputed A, b (O(N^2) per step)."""
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


def run_glauber_fast(h0_true, A, b, c, N, eta, rho, alpha, T, seed, K_flip=5):
    """
    Single-replica zero-temp Glauber with precomputed stats.
    Returns Hamming distance to true mask.
    """
    rng = np.random.default_rng(seed)

    h = np.ones(N, dtype=float)
    w = rng.normal(0, 0.1, N)
    w = optimize_w_fast(w, h, A, b, eta, K=50)

    for t in range(T):
        order = rng.permutation(N)
        flipped = False
        E_curr = energy_fast(w, h, A, b, c, eta, rho, alpha)

        for j in order:
            # Flip in place (avoid copy)
            old_hj = h[j]
            h[j] = 1.0 - old_hj
            w_try = optimize_w_fast(w, h, A, b, eta, K=K_flip)
            E_try = energy_fast(w_try, h, A, b, c, eta, rho, alpha)

            if E_try < E_curr:
                w = w_try
                E_curr = E_try
                flipped = True
            else:
                h[j] = old_hj  # revert

        w = optimize_w_fast(w, h, A, b, eta, K=20)

        if not flipped and t > 2:
            break

    return hamming_distance(h, h0_true)


# ── parameters ────────────────────────────────────────────────────────────

N_VALS = [30, 60, 120, 240]
ALPHA_RATIOS = [1.0, 2.0, 4.0, 8.0]
SIGMA_VALS = [0.001, 0.01, 0.1]
ETA_VALS = [1e-5, 1e-4, 1e-3]
ALPHA = 1.0
N_SEEDS = 4
HAMMING_THRESHOLD = 0.15

# Geometric rho grid: 18 points from 1e-7 to 0.1
RHO_GRID = np.geomspace(1e-7, 0.1, 18)


# ── precompute datasets ──────────────────────────────────────────────────

def precompute_datasets(N, M, sigma, n_seeds):
    """Generate and precompute datasets for reuse across eta/rho."""
    datasets = []
    for seed in range(n_seeds):
        X, y, w0, h0 = sample_perceptron(N, M, p0=0.5, sigma=sigma, seed=seed)
        A, b, c = precompute(X, y)
        datasets.append((A, b, c, h0, N))
    return datasets


def find_rho_c(datasets, eta, alpha, threshold):
    """Sweep rho grid, return rho_c where mean hamming < threshold."""
    N = datasets[0][3].shape[0]

    if N <= 60:
        T, K_flip = 20, 5
    elif N <= 120:
        T, K_flip = 15, 5
    else:
        T, K_flip = 12, 5

    prev_rho = 0.0
    prev_hd = 0.5

    for rho in RHO_GRID:
        hds = []
        for i, (A, b, c, h0, Ni) in enumerate(datasets):
            hd = run_glauber_fast(h0, A, b, c, Ni, eta, rho, alpha, T,
                                   seed=i + 100, K_flip=K_flip)
            hds.append(hd)
        mean_hd = np.mean(hds)

        if mean_hd < threshold:
            # Log-interpolate between prev_rho and rho
            if prev_rho > 0 and prev_hd > threshold:
                frac = (prev_hd - threshold) / (prev_hd - mean_hd + 1e-12)
                log_rc = np.log(prev_rho) + frac * (np.log(rho) - np.log(prev_rho))
                rho_c = np.exp(log_rc)
            else:
                rho_c = rho
            return rho_c, mean_hd

        prev_rho = rho
        prev_hd = mean_hd

    return np.nan, prev_hd


# ── main sweep ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = []
    total = len(N_VALS) * len(ALPHA_RATIOS) * len(SIGMA_VALS) * len(ETA_VALS)
    done = 0
    t0 = time.time()

    print(f"Experiment 18: rho_c scaling law")
    print(f"Grid: {len(N_VALS)} N × {len(ALPHA_RATIOS)} alpha_ratio × "
          f"{len(SIGMA_VALS)} sigma × {len(ETA_VALS)} eta = {total} combos")
    print(f"Seeds: {N_SEEDS}, Rho grid: {len(RHO_GRID)} points [{RHO_GRID[0]:.1e}, {RHO_GRID[-1]:.1e}]")
    print(f"Using precomputed X^TX/M for O(N^2) inner loop")
    print()

    for N in N_VALS:
        for alpha_ratio in ALPHA_RATIOS:
            M = int(N * alpha_ratio)

            for sigma in SIGMA_VALS:
                datasets = precompute_datasets(N, M, sigma, N_SEEDS)

                for eta in ETA_VALS:
                    rho_c_emp, hd_at_rc = find_rho_c(
                        datasets, eta, ALPHA, HAMMING_THRESHOLD
                    )
                    rho_c_theory = 2.0 * np.sqrt(ALPHA * eta)
                    ratio = rho_c_emp / rho_c_theory if (rho_c_theory > 0 and not np.isnan(rho_c_emp)) else np.nan

                    row = dict(
                        N=N, M=M, alpha_ratio=alpha_ratio,
                        sigma=sigma, eta=eta,
                        rho_c_empirical=rho_c_emp,
                        rho_c_theory=rho_c_theory,
                        ratio=ratio
                    )
                    results.append(row)
                    done += 1
                    elapsed = time.time() - t0
                    eta_rem = (elapsed / done) * (total - done) if done > 0 else 0

                    rho_str = f"{rho_c_emp:.6f}" if not np.isnan(rho_c_emp) else "NaN"
                    ratio_str = f"{ratio:.3f}" if not np.isnan(ratio) else "NaN"
                    print(f"[{done}/{total}] N={N} M/N={alpha_ratio:.0f} "
                          f"σ={sigma} η={eta:.0e}  "
                          f"ρ_c={rho_str}  thy={rho_c_theory:.6f}  "
                          f"r={ratio_str}  "
                          f"({elapsed:.0f}s, ~{eta_rem:.0f}s left)",
                          flush=True)

    # ── save CSV ──────────────────────────────────────────────────────────

    os.makedirs('/home/petty/pruning-research/results', exist_ok=True)
    csv_path = '/home/petty/pruning-research/results/rho_c_scaling.csv'
    fieldnames = ['N', 'M', 'alpha_ratio', 'sigma', 'eta',
                  'rho_c_empirical', 'rho_c_theory', 'ratio']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} rows to {csv_path}")

    # ── power law fit ─────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("POWER LAW FIT: rho_c ~ C * N^a * (M/N)^b * sigma^c * eta^d")
    print("="*60)

    valid = [r for r in results if not np.isnan(r['rho_c_empirical']) and r['rho_c_empirical'] > 0]
    print(f"Valid data points: {len(valid)} / {len(results)}")

    if len(valid) >= 5:
        log_rho_c = np.array([np.log(r['rho_c_empirical']) for r in valid])
        log_N = np.array([np.log(r['N']) for r in valid])
        log_alpha = np.array([np.log(r['alpha_ratio']) for r in valid])
        log_sigma = np.array([np.log(r['sigma']) for r in valid])
        log_eta = np.array([np.log(r['eta']) for r in valid])

        A_mat = np.column_stack([np.ones(len(valid)), log_N, log_alpha, log_sigma, log_eta])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_rho_c, rcond=None)
        log_C, a, b_coeff, c_coeff, d = coeffs

        y_pred = A_mat @ coeffs
        ss_res = np.sum((log_rho_c - y_pred) ** 2)
        ss_tot = np.sum((log_rho_c - np.mean(log_rho_c)) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        C = np.exp(log_C)

        print(f"\nFitted: rho_c ≈ {C:.6f} × N^{a:.3f} × (M/N)^{b_coeff:.3f} × sigma^{c_coeff:.3f} × eta^{d:.3f}")
        print(f"R² = {R2:.4f}")
        print(f"\nExponent breakdown:")
        print(f"  N exponent (a):       {a:.3f}")
        print(f"  M/N exponent (b):     {b_coeff:.3f}")
        print(f"  sigma exponent (c):   {c_coeff:.3f}")
        print(f"  eta exponent (d):     {d:.3f}")
        print(f"  Prefactor C:          {C:.6f}")
        print(f"\nTheory: rho_c ≈ 2*sqrt(alpha*eta) → eta^0.5, no N/sigma dependence")
        print(f"  eta:   {d:.3f} vs 0.5")
        print(f"  N:     {a:.3f} vs 0.0")
        print(f"  sigma: {c_coeff:.3f} vs 0.0")
        print(f"  M/N:   {b_coeff:.3f} vs 0.0")

        # Save fit summary
        fit_path = '/home/petty/pruning-research/results/rho_c_scaling_fit.txt'
        with open(fit_path, 'w') as fout:
            fout.write("Experiment 18: rho_c Scaling Law Fit\n")
            fout.write("=" * 50 + "\n\n")
            fout.write(f"Model: rho_c ~ C × N^a × (M/N)^b × sigma^c × eta^d\n\n")
            fout.write(f"Fitted parameters:\n")
            fout.write(f"  C       = {C:.6f}\n")
            fout.write(f"  a (N)   = {a:.3f}\n")
            fout.write(f"  b (M/N) = {b_coeff:.3f}\n")
            fout.write(f"  c (σ)   = {c_coeff:.3f}\n")
            fout.write(f"  d (η)   = {d:.3f}\n\n")
            fout.write(f"R² = {R2:.4f}\n")
            fout.write(f"Valid data points: {len(valid)} / {len(results)}\n\n")
            fout.write(f"Theory comparison (Mozeika: rho_c ≈ 2√(α·η)):\n")
            fout.write(f"  η exponent:   {d:.3f}  (theory: 0.5)\n")
            fout.write(f"  N exponent:   {a:.3f}  (theory: 0.0)\n")
            fout.write(f"  σ exponent:   {c_coeff:.3f}  (theory: 0.0)\n")
            fout.write(f"  M/N exponent: {b_coeff:.3f}  (theory: 0.0)\n\n")

            exponents = [('η', d, abs(d)), ('σ', c_coeff, abs(c_coeff)),
                         ('N', a, abs(a)), ('M/N', b_coeff, abs(b_coeff))]
            exponents.sort(key=lambda x: x[2], reverse=True)

            fout.write("Interpretation:\n")
            fout.write(f"  Parameters ranked by importance (|exponent|):\n")
            for name, val, absval in exponents:
                tag = "STRONG" if absval > 0.3 else ("moderate" if absval > 0.1 else "weak")
                fout.write(f"    {name}: {val:+.3f} ({tag})\n")

            fout.write(f"\n  σ relevance: {'σ matters — noise affects rho_c' if abs(c_coeff) > 0.1 else 'σ negligible at these noise levels'}\n")
            fout.write(f"  N scaling: {'rho_c depends on N' if abs(a) > 0.1 else 'rho_c intensive (N-independent) — agrees with theory'}\n")

        print(f"\nFit summary saved to {fit_path}")

        # Store for interpretation section
        fit_exponents = exponents
        fit_d = d
        fit_c = c_coeff
        fit_a = a
        fit_b = b_coeff
    else:
        print("Not enough valid data points for regression.")
        fit_exponents = None

    # ── comparison table ──────────────────────────────────────────────────

    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(f"{'N':>5} {'M/N':>5} {'sigma':>8} {'eta':>8} {'rho_c_emp':>12} {'rho_c_thy':>12} {'ratio':>8}")
    print("-" * 70)
    for r in results:
        rho_emp_str = f"{r['rho_c_empirical']:.6f}" if not np.isnan(r['rho_c_empirical']) else "NaN"
        ratio_str = f"{r['ratio']:.3f}" if not np.isnan(r['ratio']) else "NaN"
        print(f"{r['N']:>5} {r['alpha_ratio']:>5.1f} {r['sigma']:>8.3f} {r['eta']:>8.1e} "
              f"{rho_emp_str:>12} {r['rho_c_theory']:>12.6f} {ratio_str:>8}")

    # ── interpretation ────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    if fit_exponents is not None:
        print(f"\n1. Dominant parameters:")
        for name, val, absval in fit_exponents:
            marker = " ← STRONG" if absval > 0.3 else (" ← moderate" if absval > 0.1 else " (weak)")
            print(f"   {name}: exponent = {val:+.3f}{marker}")

        print(f"\n2. Theory comparison:")
        print(f"   Mozeika formula: rho_c = 2√(α·η)")
        if abs(fit_d - 0.5) < 0.15:
            print(f"   η exponent ({fit_d:.3f}) ≈ 0.5 — theory captures dominant scaling")
        else:
            print(f"   η exponent ({fit_d:.3f}) ≠ 0.5 — theory misses important physics")

        if abs(fit_c) > 0.1:
            print(f"\n3. Noise matters: σ^{fit_c:.2f}")
            print(f"   Noise raises per-weight loss, requiring higher ρ to overcome.")
            print(f"   At small σ (σ²/M ≪ signal/N), this becomes negligible.")
        else:
            print(f"\n3. Noise negligible (σ exponent {fit_c:.3f})")
            print(f"   In the low-noise regime, signal dominates — consistent with theory.")

        if abs(fit_a) > 0.1:
            print(f"\n4. System size matters: N^{fit_a:.2f}")
            print(f"   rho_c is NOT intensive — deviates from large-N theory.")
        else:
            print(f"\n4. System size negligible (N exponent {fit_a:.3f})")
            print(f"   rho_c is intensive — consistent with thermodynamic limit.")

        if abs(fit_b) > 0.1:
            print(f"\n5. Data ratio matters: (M/N)^{fit_b:.2f}")
            print(f"   More data changes the transition — not captured by Mozeika formula.")
        else:
            print(f"\n5. Data ratio negligible ((M/N) exponent {fit_b:.3f})")

    print(f"\nTotal time: {time.time() - t0:.0f}s")
