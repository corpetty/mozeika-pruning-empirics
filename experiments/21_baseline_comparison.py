"""
Experiment 21: Mozeika vs Baseline Pruning Methods

Rigorous apples-to-apples comparison of Mozeika-guided pruning vs magnitude
pruning, L1 regularization, and random pruning at matched sparsity levels
on a 2-layer MLP.

Architecture: 64 inputs -> 32 hidden (ReLU) -> 1 output
Task: Synthetic regression y = ReLU(X @ W1) @ w2 + noise (sigma=0.05)
Sparsity levels: [0%, 25%, 50%, 65%, 75%, 85%, 90%]
Seeds: 8 per method x sparsity

Methods:
  1. Mozeika (GlauberPruner) — rho grid scan, pick closest sparsity
  2. Magnitude pruning — train then prune smallest |w|
  3. Magnitude + retrain — prune then retrain 100 steps
  4. L1 regularization — train with L1 penalty, tune lambda for target sparsity
  5. Random pruning — train then randomly zero out weights
"""

import numpy as np
import sys, csv, os, time

sys.path.insert(0, '/home/petty/pruning-research')


# ── Data generation ──────────────────────────────────────────────────────

def generate_mlp_data(N_in, N_hid, M, sigma, rng):
    """Generate synthetic MLP regression data: y = ReLU(X @ W1) @ w2 + noise"""
    X = rng.standard_normal((M, N_in)) / np.sqrt(N_in)
    W1_true = rng.standard_normal((N_in, N_hid)) * 0.5
    w2_true = rng.standard_normal((N_hid, 1)) * 0.5
    hidden = np.maximum(0, X @ W1_true)
    y = (hidden @ w2_true).flatten() + sigma * rng.standard_normal(M)
    return X, y


# ── MLP helpers ──────────────────────────────────────────────────────────

def mlp_forward_flat(params, X, N_in, N_hid):
    """Forward pass from flat parameter vector."""
    split = N_in * N_hid
    W1 = params[:split].reshape(N_in, N_hid)
    w2 = params[split:].reshape(N_hid, 1)
    hidden = np.maximum(0, X @ W1)
    return (hidden @ w2).flatten(), hidden


def mlp_grad(params, X, y, N_in, N_hid, mask=None):
    """Gradient of MSE w.r.t. params (with optional mask applied)."""
    M = X.shape[0]
    split = N_in * N_hid

    p = params * mask if mask is not None else params

    W1 = p[:split].reshape(N_in, N_hid)
    w2 = p[split:].reshape(N_hid, 1)

    z1 = X @ W1
    h1 = np.maximum(0, z1)
    pred = (h1 @ w2).flatten()

    resid = (pred - y) / M
    grad_w2 = h1.T @ resid.reshape(-1, 1)
    delta = resid.reshape(-1, 1) @ w2.T
    delta = delta * (z1 > 0).astype(float)
    grad_W1 = X.T @ delta

    grad = np.concatenate([grad_W1.flatten(), grad_w2.flatten()])
    if mask is not None:
        grad = grad * mask
    return grad


def train_adam(params, X, y, N_in, N_hid, K=300, lr=0.01,
               mask=None, l1_lambda=0.0):
    """Train with Adam optimizer. Optional mask and proximal L1 penalty."""
    p = params.copy()
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    beta1, beta2, eps = 0.9, 0.99, 1e-8

    for k in range(1, K + 1):
        grad = mlp_grad(p, X, y, N_in, N_hid, mask=mask)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** k)
        v_hat = v / (1 - beta2 ** k)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # Proximal step for L1: soft thresholding
        if l1_lambda > 0:
            threshold = lr * l1_lambda
            p = np.sign(p) * np.maximum(np.abs(p) - threshold, 0.0)

        if mask is not None:
            p = p * mask

    return p


def count_sparsity(params, tol=1e-8):
    """Fraction of zero (or near-zero) weights."""
    return float(np.mean(np.abs(params) < tol))


# ── Mozeika (Glauber) method ────────────────────────────────────────────

def mozeika_energy(params, mask, X, y, N_in, N_hid, eta, rho, alpha=1.0):
    """Total Mozeika energy for flat MLP params."""
    p_masked = params * mask
    pred, _ = mlp_forward_flat(p_masked, X, N_in, N_hid)
    L = np.mean((pred - y) ** 2) / 2
    reg = (eta / 2) * np.sum(params ** 2)
    V = alpha * np.sum(mask**2 * (mask - 1)**2) + (rho / 2) * np.sum(mask)
    return L + reg + V


def mozeika_train_w(params, mask, X, y, N_in, N_hid, eta, K=30, lr=0.01):
    """Adam on weights with fixed binary mask and L2 reg."""
    p = params.copy()
    m_adam = np.zeros_like(p)
    v_adam = np.zeros_like(p)

    for k in range(1, K + 1):
        grad = mlp_grad(p, X, y, N_in, N_hid, mask=mask)
        grad = grad + eta * p

        m_adam = 0.9 * m_adam + 0.1 * grad
        v_adam = 0.99 * v_adam + 0.01 * grad ** 2
        m_hat = m_adam / (1 - 0.9 ** k)
        v_hat = v_adam / (1 - 0.99 ** k)
        p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        p = p * mask

    return p


def mozeika_glauber(X, y, N_in, N_hid, eta, rho, alpha, T_sweeps, rng,
                    K_flip=2, K_init=30):
    """Run Glauber dynamics on flat MLP params. Returns (params, mask)."""
    N_params = N_in * N_hid + N_hid

    params = rng.standard_normal(N_params) * 0.1
    mask = np.ones(N_params, dtype=float)

    params = mozeika_train_w(params, mask, X, y, N_in, N_hid, eta, K=K_init)

    for t in range(T_sweeps):
        order = rng.permutation(N_params)
        flipped = False
        E_curr = mozeika_energy(params, mask, X, y, N_in, N_hid, eta, rho, alpha)

        for j in order:
            old_mj = mask[j]
            mask[j] = 1.0 - old_mj

            p_try = mozeika_train_w(params, mask, X, y, N_in, N_hid, eta,
                                    K=K_flip)
            E_try = mozeika_energy(p_try, mask, X, y, N_in, N_hid, eta, rho, alpha)

            if E_try < E_curr:
                params = p_try
                E_curr = E_try
                flipped = True
            else:
                mask[j] = old_mj

        params = mozeika_train_w(params, mask, X, y, N_in, N_hid, eta, K=15)

        if not flipped and t > 0:
            break

    return params * mask, mask


def run_mozeika_rho_grid(X_train, y_train, N_in, N_hid, rng):
    """Run Glauber at a grid of rho values, return list of (sparsity, params).

    This is much more efficient than binary searching rho per target sparsity,
    because one pass covers all sparsity levels.
    """
    eta = 0.0001
    alpha = 1.0
    T_sweeps = 3

    # Dense rho grid spanning the full sparsity range.
    # The transition is sharp, so we need many points near it.
    rho_grid = np.geomspace(5e-5, 0.3, 20).tolist()

    scan_results = []
    rng_state = rng.bit_generator.state

    for rho in rho_grid:
        rng.bit_generator.state = rng_state  # same init for fair comparison

        p_pruned, mask = mozeika_glauber(
            X_train, y_train, N_in, N_hid, eta, rho, alpha, T_sweeps, rng,
            K_flip=2, K_init=30
        )
        sp = count_sparsity(p_pruned)
        scan_results.append((sp, p_pruned.copy(), rho))

    return scan_results


# ── Baseline method implementations ─────────────────────────────────────

def run_baselines_for_seed(X_train, y_train, X_test, y_test, N_in, N_hid,
                           target_sparsities, rng):
    """Run all baseline methods for one seed, all sparsity levels.

    Returns dict: method_name -> [(target_sp, actual_sp, test_mse), ...]
    """
    N_params = N_in * N_hid + N_hid
    results = {m: [] for m in ['Magnitude', 'Mag+Retrain', 'Random']}

    # Train once (shared across magnitude/mag+retrain/random)
    p0 = rng.standard_normal(N_params) * 0.1
    p_trained = train_adam(p0, X_train, y_train, N_in, N_hid, K=300, lr=0.01)

    for target_sp in target_sparsities:
        if target_sp == 0.0:
            pred, _ = mlp_forward_flat(p_trained, X_test, N_in, N_hid)
            mse = float(np.mean((pred - y_test) ** 2))
            for m in results:
                results[m].append((0.0, 0.0, mse))
            continue

        n_prune = int(round(target_sp * N_params))

        # Magnitude pruning
        indices = np.argsort(np.abs(p_trained))
        p_mag = p_trained.copy()
        p_mag[indices[:n_prune]] = 0.0
        pred, _ = mlp_forward_flat(p_mag, X_test, N_in, N_hid)
        results['Magnitude'].append((target_sp, count_sparsity(p_mag),
                                     float(np.mean((pred - y_test) ** 2))))

        # Magnitude + retrain
        mask = (p_mag != 0).astype(float)
        p_retrained = train_adam(p_mag.copy(), X_train, y_train, N_in, N_hid,
                                 K=100, lr=0.01, mask=mask)
        pred, _ = mlp_forward_flat(p_retrained, X_test, N_in, N_hid)
        results['Mag+Retrain'].append((target_sp, count_sparsity(p_retrained),
                                       float(np.mean((pred - y_test) ** 2))))

        # Random pruning
        rand_indices = rng.choice(N_params, size=n_prune, replace=False)
        p_rand = p_trained.copy()
        p_rand[rand_indices] = 0.0
        pred, _ = mlp_forward_flat(p_rand, X_test, N_in, N_hid)
        results['Random'].append((target_sp, count_sparsity(p_rand),
                                  float(np.mean((pred - y_test) ** 2))))

    return results


def run_l1_for_seed(X_train, y_train, X_test, y_test, N_in, N_hid,
                    target_sparsities, rng):
    """Run L1 method for one seed, all sparsity levels."""
    N_params = N_in * N_hid + N_hid
    results = []

    # For 0% sparsity: just train without L1
    p0_base = rng.standard_normal(N_params) * 0.1
    p_dense = train_adam(p0_base.copy(), X_train, y_train, N_in, N_hid,
                         K=300, lr=0.01)
    pred, _ = mlp_forward_flat(p_dense, X_test, N_in, N_hid)
    dense_mse = float(np.mean((pred - y_test) ** 2))

    for target_sp in target_sparsities:
        if target_sp == 0.0:
            results.append((0.0, 0.0, dense_mse))
            continue

        # Binary search on lambda (proximal L1 drives weights to exact zero)
        lam_lo, lam_hi = 1e-5, 10.0
        best_params = None
        best_diff = float('inf')
        best_sp = 0.0

        for _ in range(18):
            lam_mid = np.sqrt(lam_lo * lam_hi)
            p_trained = train_adam(p0_base.copy(), X_train, y_train, N_in, N_hid,
                                   K=300, lr=0.01, l1_lambda=lam_mid)
            sp = count_sparsity(p_trained)

            diff = abs(sp - target_sp)
            if diff < best_diff:
                best_diff = diff
                best_params = p_trained.copy()
                best_sp = sp

            if sp < target_sp:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid

            if diff < 0.02:
                break

        pred, _ = mlp_forward_flat(best_params, X_test, N_in, N_hid)
        mse = float(np.mean((pred - y_test) ** 2))
        results.append((target_sp, best_sp, mse))

    return results


# ── Main experiment ──────────────────────────────────────────────────────

N_IN = 64
N_HID = 32
M_TOTAL = 512
SIGMA = 0.05
SPARSITY_LEVELS = [0.0, 0.25, 0.50, 0.65, 0.75, 0.85, 0.90]
N_SEEDS = 8


if __name__ == '__main__':
    t0 = time.time()

    print("=" * 70)
    print("Experiment 21: Mozeika vs Baseline Pruning Methods")
    print("=" * 70)
    print(f"Architecture: {N_IN} -> {N_HID} (ReLU) -> 1")
    print(f"Data: M={M_TOTAL}, sigma={SIGMA}")
    print(f"Sparsity levels: {SPARSITY_LEVELS}")
    print(f"Seeds: {N_SEEDS}")
    print()

    M_train = int(M_TOTAL * 0.8)
    M_test = M_TOTAL - M_train

    # Pre-generate datasets
    datasets = {}
    for seed in range(N_SEEDS):
        rng_data = np.random.default_rng(seed * 1000)
        X, y = generate_mlp_data(N_IN, N_HID, M_TOTAL, SIGMA, rng_data)
        datasets[seed] = (X[:M_train], y[:M_train], X[M_train:], y[M_train:])

    # Collect all results: method -> {target_sp -> [mse_per_seed]}
    all_results = {m: {sp: [] for sp in SPARSITY_LEVELS}
                   for m in ['Mozeika', 'Magnitude', 'Mag+Retrain', 'L1', 'Random']}
    all_sparsities = {m: {sp: [] for sp in SPARSITY_LEVELS}
                      for m in ['Mozeika', 'Magnitude', 'Mag+Retrain', 'L1', 'Random']}

    # ── Run baselines (fast) ─────────────────────────────────────────────

    print("--- Running baselines (Magnitude, Mag+Retrain, Random) ---")
    for seed in range(N_SEEDS):
        X_tr, y_tr, X_te, y_te = datasets[seed]
        rng = np.random.default_rng(seed * 100 + 42)

        baseline_res = run_baselines_for_seed(
            X_tr, y_tr, X_te, y_te, N_IN, N_HID, SPARSITY_LEVELS, rng
        )
        for method_name, entries in baseline_res.items():
            for i, (tsp, asp, mse) in enumerate(entries):
                sp_key = SPARSITY_LEVELS[i]
                all_results[method_name][sp_key].append(mse)
                all_sparsities[method_name][sp_key].append(asp)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s", flush=True)

    # ── Run L1 ───────────────────────────────────────────────────────────

    print("\n--- Running L1 ---")
    for seed in range(N_SEEDS):
        X_tr, y_tr, X_te, y_te = datasets[seed]
        rng = np.random.default_rng(seed * 100 + 42)

        l1_res = run_l1_for_seed(
            X_tr, y_tr, X_te, y_te, N_IN, N_HID, SPARSITY_LEVELS, rng
        )
        for i, (tsp, asp, mse) in enumerate(l1_res):
            sp_key = SPARSITY_LEVELS[i]
            all_results['L1'][sp_key].append(mse)
            all_sparsities['L1'][sp_key].append(asp)

        print(f"  seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    # ── Run Mozeika (slowest) ────────────────────────────────────────────

    print("\n--- Running Mozeika (Glauber rho-grid scan) ---")
    for seed in range(N_SEEDS):
        X_tr, y_tr, X_te, y_te = datasets[seed]
        rng = np.random.default_rng(seed * 100 + 42)

        # Run rho grid scan once per seed
        scan_results = run_mozeika_rho_grid(X_tr, y_tr, N_IN, N_HID, rng)

        # For each target sparsity, pick closest rho result
        for target_sp in SPARSITY_LEVELS:
            if target_sp == 0.0:
                # Dense: just train normally
                N_params = N_IN * N_HID + N_HID
                rng_dense = np.random.default_rng(seed * 100 + 42)
                p0 = rng_dense.standard_normal(N_params) * 0.1
                p_dense = train_adam(p0, X_tr, y_tr, N_IN, N_HID, K=300, lr=0.01)
                pred, _ = mlp_forward_flat(p_dense, X_te, N_IN, N_HID)
                mse = float(np.mean((pred - y_te) ** 2))
                all_results['Mozeika'][0.0].append(mse)
                all_sparsities['Mozeika'][0.0].append(0.0)
                continue

            # Pick the scan result closest to target sparsity
            best_idx = min(range(len(scan_results)),
                           key=lambda i: abs(scan_results[i][0] - target_sp))
            sp_actual, p_pruned, rho_used = scan_results[best_idx]

            pred, _ = mlp_forward_flat(p_pruned, X_te, N_IN, N_HID)
            mse = float(np.mean((pred - y_te) ** 2))
            all_results['Mozeika'][target_sp].append(mse)
            all_sparsities['Mozeika'][target_sp].append(sp_actual)

        print(f"  seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    # ── Save CSV ─────────────────────────────────────────────────────────

    os.makedirs('/home/petty/pruning-research/results', exist_ok=True)
    csv_path = '/home/petty/pruning-research/results/baseline_comparison.csv'

    rows = []
    method_names = ['Mozeika', 'Magnitude', 'Mag+Retrain', 'L1', 'Random']
    for method in method_names:
        for sp in SPARSITY_LEVELS:
            mses = all_results[method][sp]
            sps = all_sparsities[method][sp]
            rows.append({
                'method': method,
                'target_sparsity': sp,
                'actual_sparsity': float(np.mean(sps)),
                'test_mse_mean': float(np.mean(mses)),
                'test_mse_std': float(np.std(mses)),
            })

    fieldnames = ['method', 'target_sparsity', 'actual_sparsity',
                  'test_mse_mean', 'test_mse_std']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Per-seed detail CSV
    csv_detail = '/home/petty/pruning-research/results/baseline_comparison_detail.csv'
    detail_rows = []
    for method in method_names:
        for sp in SPARSITY_LEVELS:
            for seed_i, (mse, asp) in enumerate(zip(
                    all_results[method][sp], all_sparsities[method][sp])):
                detail_rows.append({
                    'method': method, 'target_sparsity': sp,
                    'actual_sparsity': asp, 'test_mse_mean': mse,
                    'test_mse_std': 0.0, 'seed': seed_i,
                })

    with open(csv_detail, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + ['seed'])
        writer.writeheader()
        writer.writerows(detail_rows)

    print(f"\nSaved {len(rows)} summary rows to {csv_path}")
    print(f"Saved {len(detail_rows)} detail rows to {csv_detail}")

    # ── Summary table ────────────────────────────────────────────────────

    print("\n" + "=" * 85)
    print("SUMMARY TABLE — Test MSE (lower = better)")
    print("=" * 85)

    lookup = {(r['method'], r['target_sparsity']): r for r in rows}

    header = f"{'Sparsity':>10}"
    for m in method_names:
        header += f"  {m:>12}"
    print(header)
    print("-" * len(header))

    for sp in SPARSITY_LEVELS:
        row_str = f"{sp:>10.0%}"
        for m in method_names:
            r = lookup.get((m, sp))
            if r:
                row_str += f"  {r['test_mse_mean']:>12.6f}"
            else:
                row_str += f"  {'N/A':>12}"
        print(row_str)

    # Actual sparsity achieved
    print(f"\n{'Actual sparsity achieved':>10}")
    header2 = f"{'Target':>10}"
    for m in method_names:
        header2 += f"  {m:>12}"
    print(header2)
    print("-" * len(header2))
    for sp in SPARSITY_LEVELS:
        row_str = f"{sp:>10.0%}"
        for m in method_names:
            r = lookup.get((m, sp))
            if r:
                row_str += f"  {r['actual_sparsity']:>12.3f}"
            else:
                row_str += f"  {'N/A':>12}"
        print(row_str)

    # ── Analysis ─────────────────────────────────────────────────────────

    print("\n" + "=" * 85)
    print("ANALYSIS")
    print("=" * 85)

    dense_mses = [lookup.get((m, 0.0), {}).get('test_mse_mean', float('nan'))
                  for m in method_names]
    dense_baseline = float(np.nanmean(dense_mses))
    print(f"\nDense baseline MSE (0% sparsity, avg): {dense_baseline:.6f}")

    print("\nDivergence point (MSE > 2x dense baseline):")
    for m in method_names:
        diverged = False
        for sp in SPARSITY_LEVELS[1:]:
            r = lookup.get((m, sp))
            if r and r['test_mse_mean'] > 2 * dense_baseline:
                print(f"  {m:>12}: diverges at {sp:.0%} "
                      f"(MSE={r['test_mse_mean']:.6f})")
                diverged = True
                break
        if not diverged:
            print(f"  {m:>12}: never diverges (all < 2x baseline)")

    # Head-to-head: Mozeika vs Magnitude+Retrain
    print("\nMozeika vs Magnitude+Retrain (head-to-head):")
    mozeika_wins = 0
    mag_retrain_wins = 0
    ties = 0

    for sp in SPARSITY_LEVELS[1:]:
        moz = lookup.get(('Mozeika', sp), {}).get('test_mse_mean', float('nan'))
        mag_r = lookup.get(('Mag+Retrain', sp), {}).get('test_mse_mean', float('nan'))

        if np.isnan(moz) or np.isnan(mag_r):
            continue

        ratio = moz / mag_r if mag_r > 0 else float('inf')

        if ratio < 0.95:
            winner = "MOZEIKA"
            mozeika_wins += 1
        elif ratio > 1.05:
            winner = "MAG+RETRAIN"
            mag_retrain_wins += 1
        else:
            winner = "TIE"
            ties += 1

        print(f"  {sp:>5.0%}: Mozeika={moz:.6f} vs Mag+Retrain={mag_r:.6f} "
              f"(ratio={ratio:.3f}) -> {winner}")

    print(f"\nScore: Mozeika={mozeika_wins}, Mag+Retrain={mag_retrain_wins}, "
          f"Ties={ties}")

    if mozeika_wins >= 3:
        verdict = "MOZEIKA WINS"
    elif mag_retrain_wins >= 3:
        verdict = "MAGNITUDE WINS"
    else:
        verdict = "TIE"

    print(f"\n{'='*40}")
    print(f"  VERDICT: {verdict}")
    print(f"{'='*40}")

    print(f"\nTotal time: {time.time() - t0:.0f}s")
