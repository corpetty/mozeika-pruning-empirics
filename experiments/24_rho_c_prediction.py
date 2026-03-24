"""
Experiment 24: rho_c prediction accuracy.

Goal: Can we predict rho_c *before* running the full sweep, using only
network architecture + data statistics?

Two predictors:
  1. Mozeika formula: rho_c = 2*sqrt(alpha*eta)
  2. Empirical fit (Exp 18): rho_c = 0.043 * N^(-0.65) * (M/N)^(-0.83) * sigma^0.37 * eta^0.24

Method:
  - Generate 50 random parameter combinations
  - For each: run Glauber rho sweep (6 seeds) to get true rho_c
  - Compute both predictions
  - Metrics: correlation, MAE, RMSE, fraction within 2x of true value
"""

import numpy as np
import sys, csv, os, time

sys.path.insert(0, '/home/petty/pruning-research')

from pruning_core.data import sample_perceptron
from pruning_core.metrics import hamming_distance


# ── precomputed helpers (from Exp 18) ────────────────────────────────────

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


def run_glauber_fast(h0_true, A, b, c, N, eta, rho, alpha, T, seed, K_flip=5):
    """Zero-temp Glauber with precomputed stats. Returns Hamming."""
    rng = np.random.default_rng(seed)
    h = np.ones(N, dtype=float)
    w = rng.normal(0, 0.1, N)
    w = optimize_w_fast(w, h, A, b, eta, K=50)

    for t in range(T):
        order = rng.permutation(N)
        flipped = False
        E_curr = energy_fast(w, h, A, b, c, eta, rho, alpha)

        for j in order:
            old_hj = h[j]
            h[j] = 1.0 - old_hj
            w_try = optimize_w_fast(w, h, A, b, eta, K=K_flip)
            E_try = energy_fast(w_try, h, A, b, c, eta, rho, alpha)

            if E_try < E_curr:
                w = w_try
                E_curr = E_try
                flipped = True
            else:
                h[j] = old_hj

        w = optimize_w_fast(w, h, A, b, eta, K=20)
        if not flipped and t > 2:
            break

    return hamming_distance(h, h0_true)


# ── rho_c finder ─────────────────────────────────────────────────────────

HAMMING_THRESHOLD = 0.15

def find_rho_c(datasets, N, eta, alpha=1.0):
    """Sweep geometric rho grid, return rho_c via log-interpolation."""
    rho_grid = np.geomspace(1e-7, 0.1, 20)

    if N <= 60:
        T, K_flip = 20, 5
    elif N <= 120:
        T, K_flip = 15, 5
    else:
        T, K_flip = 12, 5

    prev_rho = 0.0
    prev_hd = 0.5

    for rho in rho_grid:
        hds = []
        for i, (A, b, c, h0) in enumerate(datasets):
            hd = run_glauber_fast(h0, A, b, c, N, eta, rho, alpha, T,
                                   seed=i + 100, K_flip=K_flip)
            hds.append(hd)
        mean_hd = np.mean(hds)

        if mean_hd < HAMMING_THRESHOLD:
            if prev_rho > 0 and prev_hd > HAMMING_THRESHOLD:
                frac = (prev_hd - HAMMING_THRESHOLD) / (prev_hd - mean_hd + 1e-12)
                log_rc = np.log(prev_rho) + frac * (np.log(rho) - np.log(prev_rho))
                return np.exp(log_rc)
            return rho

        prev_rho = rho
        prev_hd = mean_hd

    return np.nan


# ── predictors ───────────────────────────────────────────────────────────

def rho_c_mozeika(alpha_ratio, eta, alpha=1.0):
    """Mozeika formula: rho_c = 2*sqrt(alpha*eta)."""
    return 2.0 * np.sqrt(alpha * eta)


def rho_c_empirical_fit(N, alpha_ratio, sigma, eta):
    """Empirical fit from Exp 18: rho_c = C * N^a * (M/N)^b * sigma^c * eta^d."""
    C = 0.042516
    a = -0.649
    b = -0.832
    c_exp = 0.365
    d = 0.235
    return C * N**a * alpha_ratio**b * sigma**c_exp * eta**d


# ── generate random parameter combos ────────────────────────────────────

N_COMBOS = 50
N_SEEDS = 6
ALPHA = 1.0

rng_params = np.random.default_rng(42)

N_choices = [30, 60, 120, 240]
alpha_choices = [1.0, 2.0, 4.0, 8.0]
sigma_choices = [0.005, 0.01, 0.05]
eta_choices = [5e-5, 1e-4, 5e-4]

combos = []
for i in range(N_COMBOS):
    combo = dict(
        N=rng_params.choice(N_choices),
        alpha_ratio=rng_params.choice(alpha_choices),
        sigma=rng_params.choice(sigma_choices),
        eta=rng_params.choice(eta_choices),
    )
    combos.append(combo)

# ── main sweep ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Experiment 24: rho_c Prediction Accuracy")
    print(f"Testing {N_COMBOS} random parameter combinations, {N_SEEDS} seeds each")
    print(f"N choices: {N_choices}")
    print(f"alpha choices: {alpha_choices}")
    print(f"sigma choices: {sigma_choices}")
    print(f"eta choices: {eta_choices}")
    print()

    results = []
    t0 = time.time()

    for idx, combo in enumerate(combos):
        N_val = int(combo['N'])
        alpha_ratio = combo['alpha_ratio']
        M_val = int(N_val * alpha_ratio)
        sigma = combo['sigma']
        eta = combo['eta']

        # Generate datasets
        datasets = []
        for seed in range(N_SEEDS):
            X, y, w0, h0 = sample_perceptron(N_val, M_val, p0=0.5,
                                               sigma=sigma, seed=seed + 1000 * idx)
            A, b, c = precompute(X, y)
            datasets.append((A, b, c, h0))

        # Find true rho_c
        rho_c_true = find_rho_c(datasets, N_val, eta, ALPHA)

        # Predictions
        rho_c_moz = rho_c_mozeika(alpha_ratio, eta, ALPHA)
        rho_c_fit = rho_c_empirical_fit(N_val, alpha_ratio, sigma, eta)

        # Errors
        if not np.isnan(rho_c_true) and rho_c_true > 0:
            err_moz = abs(rho_c_moz - rho_c_true) / rho_c_true
            err_fit = abs(rho_c_fit - rho_c_true) / rho_c_true
        else:
            err_moz = np.nan
            err_fit = np.nan

        row = dict(
            N=N_val, alpha=alpha_ratio, sigma=sigma, eta=eta,
            rho_c_true=rho_c_true,
            rho_c_mozeika=rho_c_moz,
            rho_c_fit=rho_c_fit,
            err_mozeika=err_moz,
            err_fit=err_fit,
        )
        results.append(row)

        elapsed = time.time() - t0
        eta_rem = (elapsed / (idx + 1)) * (N_COMBOS - idx - 1)
        rho_str = f"{rho_c_true:.6f}" if not np.isnan(rho_c_true) else "NaN"
        print(f"[{idx+1}/{N_COMBOS}] N={N_val} M/N={alpha_ratio:.0f} "
              f"σ={sigma} η={eta:.0e}  "
              f"ρ_c_true={rho_str}  moz={rho_c_moz:.6f}  fit={rho_c_fit:.6f}  "
              f"({elapsed:.0f}s, ~{eta_rem:.0f}s left)", flush=True)

    # ── save CSV ──────────────────────────────────────────────────────────

    out_dir = '/home/petty/pruning-research/results'
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'rho_c_prediction.csv')
    fieldnames = ['N', 'alpha', 'sigma', 'eta',
                  'rho_c_true', 'rho_c_mozeika', 'rho_c_fit',
                  'err_mozeika', 'err_fit']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows to {csv_path}")

    # ── compute metrics ──────────────────────────────────────────────────

    valid = [r for r in results if not np.isnan(r['rho_c_true']) and r['rho_c_true'] > 0]
    n_valid = len(valid)
    print(f"\nValid data points: {n_valid} / {len(results)}")

    if n_valid < 5:
        print("Not enough valid data points for metrics.")
        sys.exit(0)

    true_vals = np.array([r['rho_c_true'] for r in valid])
    moz_vals = np.array([r['rho_c_mozeika'] for r in valid])
    fit_vals = np.array([r['rho_c_fit'] for r in valid])

    def compute_metrics(pred, true):
        corr = np.corrcoef(pred, true)[0, 1]
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        # Within 2x: 0.5 <= pred/true <= 2.0
        ratio = pred / true
        within_2x = np.mean((ratio >= 0.5) & (ratio <= 2.0)) * 100
        return corr, mae, rmse, within_2x

    corr_moz, mae_moz, rmse_moz, w2x_moz = compute_metrics(moz_vals, true_vals)
    corr_fit, mae_fit, rmse_fit, w2x_fit = compute_metrics(fit_vals, true_vals)

    # ── print summary ────────────────────────────────────────────────────

    print(f"\n{'='*65}")
    print(f"{'Predictor':>20s}  {'Corr':>6s}  {'MAE':>8s}  {'RMSE':>8s}  {'Within-2x':>10s}")
    print(f"{'-'*65}")
    print(f"{'Mozeika formula':>20s}  {corr_moz:>6.3f}  {mae_moz:>8.6f}  {rmse_moz:>8.6f}  {w2x_moz:>9.0f}%")
    print(f"{'Empirical fit':>20s}  {corr_fit:>6.3f}  {mae_fit:>8.6f}  {rmse_fit:>8.6f}  {w2x_fit:>9.0f}%")
    print(f"{'='*65}")

    # ── log-space metrics (more appropriate for ratio comparison) ─────────

    log_true = np.log10(true_vals)
    log_moz = np.log10(moz_vals)
    log_fit = np.log10(fit_vals)

    corr_moz_log, mae_moz_log, rmse_moz_log, _ = compute_metrics(log_moz, log_true)
    corr_fit_log, mae_fit_log, rmse_fit_log, _ = compute_metrics(log_fit, log_true)

    print(f"\nLog-space metrics (log10):")
    print(f"{'Predictor':>20s}  {'Corr':>6s}  {'MAE':>8s}  {'RMSE':>8s}")
    print(f"{'-'*50}")
    print(f"{'Mozeika formula':>20s}  {corr_moz_log:>6.3f}  {mae_moz_log:>8.4f}  {rmse_moz_log:>8.4f}")
    print(f"{'Empirical fit':>20s}  {corr_fit_log:>6.3f}  {mae_fit_log:>8.4f}  {rmse_fit_log:>8.4f}")

    # ── detailed comparison ──────────────────────────────────────────────

    print(f"\n{'='*90}")
    print("Detailed comparison (first 20 entries):")
    print(f"{'N':>5} {'M/N':>5} {'sigma':>7} {'eta':>8} {'rho_c_true':>11} {'moz':>11} {'fit':>11} {'err_moz':>8} {'err_fit':>8}")
    print("-" * 90)
    for r in valid[:20]:
        err_m = f"{r['err_mozeika']:.2f}" if not np.isnan(r['err_mozeika']) else "NaN"
        err_f = f"{r['err_fit']:.2f}" if not np.isnan(r['err_fit']) else "NaN"
        print(f"{r['N']:>5} {r['alpha']:>5.1f} {r['sigma']:>7.3f} {r['eta']:>8.1e} "
              f"{r['rho_c_true']:>11.6f} {r['rho_c_mozeika']:>11.6f} {r['rho_c_fit']:>11.6f} "
              f"{err_m:>8} {err_f:>8}")

    # ── interpretation ───────────────────────────────────────────────────

    print(f"\n{'='*65}")
    print("INTERPRETATION")
    print(f"{'='*65}")

    if w2x_fit >= 80:
        print(f"\nEmpirical fit achieves {w2x_fit:.0f}% within-2x — USABLE for automatic")
        print("rho setting in real LLM layers without calibration sweeps.")
    elif w2x_fit >= 60:
        print(f"\nEmpirical fit achieves {w2x_fit:.0f}% within-2x — MODERATE accuracy.")
        print("Good for initial estimate, but calibration sweep still recommended.")
    else:
        print(f"\nEmpirical fit achieves {w2x_fit:.0f}% within-2x — needs improvement.")
        print("The power-law fit may not generalize well outside training range.")

    if w2x_moz < w2x_fit:
        print(f"\nEmpirical fit ({w2x_fit:.0f}%) outperforms Mozeika formula ({w2x_moz:.0f}%).")
        print("The additional N, M/N, sigma dependence matters in practice.")
    else:
        print(f"\nMozeika formula ({w2x_moz:.0f}%) competitive with empirical fit ({w2x_fit:.0f}%).")
        print("Simple 2*sqrt(alpha*eta) may be sufficient for practical use.")

    print(f"\nTotal time: {time.time() - t0:.0f}s")
