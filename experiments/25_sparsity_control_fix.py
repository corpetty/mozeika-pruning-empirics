"""
Experiment 25: Definitive Mozeika vs Baselines with Fixed Sparsity Control

Exp 21 showed Mozeika has high-sparsity stability (0.334 MSE at 90% vs
Mag+Retrain 0.811), but the rho binary search was broken:
  - Target 25% -> actual 15%
  - Target 50% -> actual 59%
  - Target 90% -> actual 90% (only OK at extremes)

FIX: After running Glauber at a given rho, measure actual achieved sparsity.
If off by more than ±2%, adjust rho and re-run. Iterate up to 8 times using
binary search on rho guided by the measured sparsity.

Architecture: 64 -> 32 (ReLU) -> 1  (same as Exp 21)
Task: Synthetic regression, sigma=0.05
Sparsity levels: [0%, 25%, 50%, 65%, 75%, 85%, 90%]
Seeds: 8 per method x sparsity
Tolerance: ±2% sparsity

THIS IS THE DEFINITIVE COMPARISON.
If Mozeika still beats Mag+Retrain at matched sparsity above 75%:
    -> the rho energy penalty has value as a pruning objective.
If not:
    -> the framework has no practical advantage. Stop.
"""

import numpy as np
import sys, csv, os, time

sys.path.insert(0, '/home/petty/pruning-research')


# ── Data generation (same as Exp 21) ────────────────────────────────────

def generate_mlp_data(N_in, N_hid, M, sigma, rng):
    """Generate synthetic MLP regression data: y = ReLU(X @ W1) @ w2 + noise"""
    X = rng.standard_normal((M, N_in)) / np.sqrt(N_in)
    W1_true = rng.standard_normal((N_in, N_hid)) * 0.5
    w2_true = rng.standard_normal((N_hid, 1)) * 0.5
    hidden = np.maximum(0, X @ W1_true)
    y = (hidden @ w2_true).flatten() + sigma * rng.standard_normal(M)
    return X, y


# ── MLP helpers (same as Exp 21) ────────────────────────────────────────

def mlp_forward_flat(params, X, N_in, N_hid):
    split = N_in * N_hid
    W1 = params[:split].reshape(N_in, N_hid)
    w2 = params[split:].reshape(N_hid, 1)
    hidden = np.maximum(0, X @ W1)
    return (hidden @ w2).flatten(), hidden


def mlp_grad(params, X, y, N_in, N_hid, mask=None):
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
        if l1_lambda > 0:
            threshold = lr * l1_lambda
            p = np.sign(p) * np.maximum(np.abs(p) - threshold, 0.0)
        if mask is not None:
            p = p * mask
    return p


def count_sparsity(params, tol=1e-8):
    return float(np.mean(np.abs(params) < tol))


# ── Mozeika (Glauber) with fixed sparsity control ──────────────────────

def mozeika_energy(params, mask, X, y, N_in, N_hid, eta, rho, alpha=1.0):
    p_masked = params * mask
    pred, _ = mlp_forward_flat(p_masked, X, N_in, N_hid)
    L = np.mean((pred - y) ** 2) / 2
    reg = (eta / 2) * np.sum(params ** 2)
    V = alpha * np.sum(mask**2 * (mask - 1)**2) + (rho / 2) * np.sum(mask)
    return L + reg + V


def mozeika_train_w(params, mask, X, y, N_in, N_hid, eta, K=30, lr=0.01):
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


def mozeika_at_target_sparsity(X, y, N_in, N_hid, target_sp, rng,
                                tol=0.02, max_iter=8):
    """
    Run Glauber with iterative rho adjustment until achieved sparsity
    is within ±tol of target_sp.

    Binary search on rho, but each iteration actually runs Glauber and
    measures the resulting sparsity. This is the fix for Exp 21's broken
    sparsity control.
    """
    eta = 0.0001
    alpha = 1.0
    T_sweeps = 3
    N_params = N_in * N_hid + N_hid

    # Initial rho bounds based on Exp 21 calibration
    rho_lo = 1e-5
    rho_hi = 0.5

    best_params = None
    best_mask = None
    best_sp = 0.0
    best_diff = float('inf')

    # Save initial RNG state so each Glauber run starts from same init
    rng_state = rng.bit_generator.state

    for iteration in range(max_iter):
        rho = np.sqrt(rho_lo * rho_hi)  # geometric midpoint

        # Reset RNG to same state for fair comparison across iterations
        rng.bit_generator.state = rng_state

        p_pruned, mask = mozeika_glauber(
            X, y, N_in, N_hid, eta, rho, alpha, T_sweeps, rng,
            K_flip=2, K_init=30
        )

        sp_actual = count_sparsity(p_pruned)
        diff = abs(sp_actual - target_sp)

        if diff < best_diff:
            best_diff = diff
            best_params = p_pruned.copy()
            best_mask = mask.copy()
            best_sp = sp_actual

        if diff <= tol:
            break

        # Adjust rho: higher rho -> more sparsity
        if sp_actual < target_sp:
            rho_lo = rho
        else:
            rho_hi = rho

    return best_params, best_mask, best_sp


# ── Baseline methods (same as Exp 21) ──────────────────────────────────

def run_baselines_for_seed(X_train, y_train, X_test, y_test, N_in, N_hid,
                           target_sparsities, rng):
    N_params = N_in * N_hid + N_hid
    results = {m: [] for m in ['Magnitude', 'Mag+Retrain', 'Random']}

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
    N_params = N_in * N_hid + N_hid
    results = []

    p0_base = rng.standard_normal(N_params) * 0.1
    p_dense = train_adam(p0_base.copy(), X_train, y_train, N_in, N_hid,
                         K=300, lr=0.01)
    pred, _ = mlp_forward_flat(p_dense, X_test, N_in, N_hid)
    dense_mse = float(np.mean((pred - y_test) ** 2))

    for target_sp in target_sparsities:
        if target_sp == 0.0:
            results.append((0.0, 0.0, dense_mse))
            continue

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


# ── Main experiment ─────────────────────────────────────────────────────

N_IN = 64
N_HID = 32
M_TOTAL = 512
SIGMA = 0.05
SPARSITY_LEVELS = [0.0, 0.25, 0.50, 0.65, 0.75, 0.85, 0.90]
N_SEEDS = 8


if __name__ == '__main__':
    t0 = time.time()

    print("=" * 70)
    print("Experiment 25: Mozeika vs Baselines — FIXED Sparsity Control")
    print("=" * 70)
    print(f"Architecture: {N_IN} -> {N_HID} (ReLU) -> 1")
    print(f"Data: M={M_TOTAL}, sigma={SIGMA}")
    print(f"Sparsity levels: {SPARSITY_LEVELS}")
    print(f"Seeds: {N_SEEDS}")
    print(f"Sparsity tolerance: ±2%")
    print(f"Max rho iterations: 8")
    print()

    M_train = int(M_TOTAL * 0.8)
    M_test = M_TOTAL - M_train

    # Pre-generate datasets
    datasets = {}
    for seed in range(N_SEEDS):
        rng_data = np.random.default_rng(seed * 1000)
        X, y = generate_mlp_data(N_IN, N_HID, M_TOTAL, SIGMA, rng_data)
        datasets[seed] = (X[:M_train], y[:M_train], X[M_train:], y[M_train:])

    all_results = {m: {sp: [] for sp in SPARSITY_LEVELS}
                   for m in ['Mozeika', 'Magnitude', 'Mag+Retrain', 'L1', 'Random']}
    all_sparsities = {m: {sp: [] for sp in SPARSITY_LEVELS}
                      for m in ['Mozeika', 'Magnitude', 'Mag+Retrain', 'L1', 'Random']}

    # ── Baselines (fast) ─────────────────────────────────────────────────

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

    print(f"  Done in {time.time() - t0:.0f}s", flush=True)

    # ── L1 ───────────────────────────────────────────────────────────────

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

    # ── Mozeika with FIXED sparsity control ──────────────────────────────

    print("\n--- Running Mozeika (iterative rho adjustment, ±2% tolerance) ---")
    for seed in range(N_SEEDS):
        X_tr, y_tr, X_te, y_te = datasets[seed]

        for target_sp in SPARSITY_LEVELS:
            if target_sp == 0.0:
                # Dense baseline
                N_params = N_IN * N_HID + N_HID
                rng_dense = np.random.default_rng(seed * 100 + 42)
                p0 = rng_dense.standard_normal(N_params) * 0.1
                p_dense = train_adam(p0, X_tr, y_tr, N_IN, N_HID, K=300, lr=0.01)
                pred, _ = mlp_forward_flat(p_dense, X_te, N_IN, N_HID)
                mse = float(np.mean((pred - y_te) ** 2))
                all_results['Mozeika'][0.0].append(mse)
                all_sparsities['Mozeika'][0.0].append(0.0)
                continue

            rng = np.random.default_rng(seed * 100 + 42)
            p_pruned, mask, sp_actual = mozeika_at_target_sparsity(
                X_tr, y_tr, N_IN, N_HID, target_sp, rng,
                tol=0.02, max_iter=8
            )
            pred, _ = mlp_forward_flat(p_pruned, X_te, N_IN, N_HID)
            mse = float(np.mean((pred - y_te) ** 2))
            all_results['Mozeika'][target_sp].append(mse)
            all_sparsities['Mozeika'][target_sp].append(sp_actual)

            print(f"  seed={seed} target={target_sp:.0%} -> actual={sp_actual:.3f} "
                  f"MSE={mse:.4f}", flush=True)

        print(f"  seed {seed} complete ({time.time() - t0:.0f}s)", flush=True)

    # ── Save CSV ─────────────────────────────────────────────────────────

    os.makedirs('/home/petty/pruning-research/results', exist_ok=True)
    csv_path = '/home/petty/pruning-research/results/baseline_comparison_fixed.csv'

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
    print(f"\nSaved {len(rows)} rows to {csv_path}")

    # ── Summary table ────────────────────────────────────────────────────

    lookup = {(r['method'], r['target_sparsity']): r for r in rows}

    print("\n" + "=" * 85)
    print("SUMMARY TABLE — Test MSE (lower = better)")
    print("=" * 85)

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
                row_str += f"  {r['test_mse_mean']:>12.4f}"
            else:
                row_str += f"  {'N/A':>12}"
        print(row_str)

    # Actual sparsity achieved
    print(f"\n{'':>10}  Actual sparsity achieved:")
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

    # ── Sparsity control quality ─────────────────────────────────────────

    print(f"\n{'='*85}")
    print("SPARSITY CONTROL QUALITY (Mozeika)")
    print(f"{'='*85}")
    max_error = 0.0
    for sp in SPARSITY_LEVELS[1:]:  # skip 0%
        r = lookup.get(('Mozeika', sp))
        if r:
            err = abs(r['actual_sparsity'] - sp)
            max_error = max(max_error, err)
            status = "OK" if err <= 0.02 else "MISS"
            print(f"  Target {sp:.0%} -> Actual {r['actual_sparsity']:.3f} "
                  f"(error={err:.3f}) [{status}]")
    print(f"  Max error: {max_error:.3f}")

    # ── Head-to-head: Mozeika vs Mag+Retrain ────────────────────────────

    print(f"\n{'='*85}")
    print("Mozeika vs Mag+Retrain (head-to-head at matched sparsity)")
    print(f"{'='*85}")

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

        moz_sp = lookup.get(('Mozeika', sp), {}).get('actual_sparsity', 0)
        print(f"  {sp:>5.0%}: Mozeika={moz:.4f} (at {moz_sp:.1%}) vs "
              f"Mag+Retrain={mag_r:.4f} -> {winner} (ratio={ratio:.3f})")

    print(f"\nScore: Mozeika={mozeika_wins}, Mag+Retrain={mag_retrain_wins}, Ties={ties}")

    # ── Also compare vs L1 at high sparsity ──────────────────────────────

    print(f"\n{'='*85}")
    print("Mozeika vs L1 at HIGH sparsity (75%, 85%, 90%)")
    print(f"{'='*85}")
    for sp in [0.75, 0.85, 0.90]:
        moz = lookup.get(('Mozeika', sp), {}).get('test_mse_mean', float('nan'))
        l1 = lookup.get(('L1', sp), {}).get('test_mse_mean', float('nan'))
        if not np.isnan(moz) and not np.isnan(l1) and l1 > 0:
            print(f"  {sp:.0%}: Mozeika={moz:.4f} vs L1={l1:.4f} "
                  f"(ratio={moz/l1:.3f})")

    # ── Final verdict ────────────────────────────────────────────────────

    print(f"\n{'='*85}")
    high_sp_wins = 0
    for sp in [0.75, 0.85, 0.90]:
        moz = lookup.get(('Mozeika', sp), {}).get('test_mse_mean', float('nan'))
        mag_r = lookup.get(('Mag+Retrain', sp), {}).get('test_mse_mean', float('nan'))
        if not np.isnan(moz) and not np.isnan(mag_r) and moz < mag_r * 0.95:
            high_sp_wins += 1

    if high_sp_wins >= 2:
        verdict = "POSITIVE: Mozeika beats Mag+Retrain at high sparsity with matched control"
        go_nogo = "GO: Proceed to GPT-2 experiment"
    else:
        verdict = "NEGATIVE: Mozeika advantage disappears with proper sparsity matching"
        go_nogo = "NO-GO: Write negative results paper and stop"

    print(f"  HIGH-SPARSITY VERDICT (wins at 75/85/90%): {high_sp_wins}/3")
    print(f"  {verdict}")
    print(f"  {go_nogo}")
    print(f"{'='*85}")

    print(f"\nTotal time: {time.time() - t0:.0f}s")
