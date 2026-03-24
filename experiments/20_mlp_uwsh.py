"""
Experiment 20: MLP Phase Transition + UWSH Jaccard Support Overlap

Tests whether:
  A) The Mozeika phase transition (sharp Hamming drop at rho_c) survives in
     a non-linear MLP [N=10 -> H=5 (tanh) -> 1].
  B) Independent runs converge to masks with overlapping *support* (Jaccard
     similarity) — the proper UWSH test.
  C) Layer-wise Hamming reveals whether layers collapse at different rho values.

Key insight: perceptron UWSH was degenerate (convex loss -> unique w*).
For non-trivial support sharing we need a non-linear MLP in an underdetermined
regime (M < N_params) so that multiple solutions exist.

Architecture: [N=10 -> H=5 (tanh) -> 1 (linear)]
Total mask entries: 10*5 + 5*1 = 55, true active = ~27 (50% per layer)

IMPORTANT: The rho scale for the MLP is much smaller than for the perceptron
because the per-weight loss contribution is O(sigma^2 / N_total_params).
With sigma=0.01, N_total=55, rho_c ~ 2*L(0) / N_active ~ 5e-5.
"""

import numpy as np
import csv, os, sys, time
from itertools import combinations

sys.path.insert(0, '/home/petty/pruning-research')

# ── MLP helpers (inline to avoid known bugs in energy_mlp.py) ─────────────

def mlp_forward(w_list, h_list, X):
    """Forward pass: tanh hidden layers, linear output."""
    a = X
    for i, (w, h) in enumerate(zip(w_list, h_list)):
        z = a @ (w * h)
        if i < len(w_list) - 1:
            a = np.tanh(z)
        else:
            a = z  # linear output
    return a.ravel()


def mlp_energy(w_list, h_list, X, y, eta, rho, alpha=1.0):
    """Total energy: MSE loss + L2 reg + double-well potential."""
    pred = mlp_forward(w_list, h_list, X)
    L = 0.5 * np.mean((pred - y) ** 2)
    reg = sum(0.5 * eta * np.sum(w ** 2) for w in w_list)
    V = 0.0
    for h in h_list:
        hf = h.ravel()
        V += np.sum(alpha * hf**2 * (hf - 1)**2 + 0.5 * rho * hf)
    return L + reg + V


def mlp_grad_w(w_list, h_list, X, y, eta):
    """Backprop gradients dE/dw_l for each layer (tanh hidden, linear output)."""
    M = X.shape[0]
    nL = len(w_list)

    activations = [X]
    zs = []
    a = X
    for i, (w, h) in enumerate(zip(w_list, h_list)):
        wm = w * h
        z = a @ wm
        zs.append(z)
        if i < nL - 1:
            a = np.tanh(z)
        else:
            a = z
        activations.append(a)

    delta = (activations[-1].ravel() - y).reshape(-1, 1) / M
    grads = []
    for l in range(nL - 1, -1, -1):
        a_l = activations[l]
        g = a_l.T @ delta
        g = g * h_list[l] + eta * w_list[l]
        grads.insert(0, g)
        if l > 0:
            wm = w_list[l] * h_list[l]
            delta = (delta @ wm.T) * (1 - np.tanh(zs[l - 1]) ** 2)
    return grads


def optimize_w_adam(w_list, h_list, X, y, eta, K=20, lr=0.01):
    """K steps of Adam on all layers."""
    ws = [w.copy() for w in w_list]
    ms = [np.zeros_like(w) for w in ws]
    vs = [np.zeros_like(w) for w in ws]
    for k in range(1, K + 1):
        grads = mlp_grad_w(ws, h_list, X, y, eta)
        for l in range(len(ws)):
            ms[l] = 0.9 * ms[l] + 0.1 * grads[l]
            vs[l] = 0.99 * vs[l] + 0.01 * grads[l] ** 2
            m_hat = ms[l] / (1 - 0.9 ** k)
            v_hat = vs[l] / (1 - 0.99 ** k)
            ws[l] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return ws


# ── Data generation ───────────────────────────────────────────────────────

def generate_mlp_data(N, H, M, sigma, rng):
    """
    Generate regression data: y = MLP(X; w_true, h_true) + noise.
    Architecture: [N -> H (tanh) -> 1 (linear)].
    True masks are 50% sparse per layer.
    """
    X = rng.standard_normal((M, N)) / np.sqrt(N)

    w0 = rng.standard_normal((N, H)) / np.sqrt(N)
    w1 = rng.standard_normal((H, 1)) / np.sqrt(H)

    h0 = np.zeros((N, H))
    for col in range(H):
        idx = rng.choice(N, size=N // 2, replace=False)
        h0[idx, col] = 1.0
    h1 = np.zeros((H, 1))
    idx = rng.choice(H, size=H // 2, replace=False)
    h1[idx, 0] = 1.0

    w_true = [w0, w1]
    h_true = [h0, h1]

    y = mlp_forward(w_true, h_true, X)
    y += sigma * rng.standard_normal(M)

    return X, y, w_true, h_true


# ── Glauber dynamics for MLP ─────────────────────────────────────────────

def mlp_glauber(X, y, h_true, eta, rho, alpha, T, seed, K_adam=20,
                return_masks=False, return_layerwise=False):
    """
    Coordinate-descent Glauber on MLP mask.
    Returns Hamming distance to true mask (and optionally the final masks
    or per-layer Hamming).
    """
    rng = np.random.default_rng(seed)

    # Init: all active, small random weights
    h = [np.ones_like(ht) for ht in h_true]
    w = [rng.normal(0, 0.1, ht.shape) for ht in h_true]

    # Initial weight optimization
    w = optimize_w_adam(w, h, X, y, eta, K=50, lr=0.01)

    # Build index list once
    all_indices = []
    for l in range(len(h)):
        for idx in range(h[l].size):
            all_indices.append((l, np.unravel_index(idx, h[l].shape)))

    for t in range(T):
        order = rng.permutation(len(all_indices))
        E_curr = mlp_energy(w, h, X, y, eta, rho, alpha)

        for oi in order:
            l, ij = all_indices[oi]
            h_try = [hh.copy() for hh in h]
            h_try[l][ij] = 1.0 - h_try[l][ij]
            w_try = optimize_w_adam(w, h_try, X, y, eta, K=K_adam, lr=0.01)
            E_try = mlp_energy(w_try, h_try, X, y, eta, rho, alpha)
            if E_try < E_curr:
                h, w, E_curr = h_try, w_try, E_try

        w = optimize_w_adam(w, h, X, y, eta, K=20, lr=0.01)

    # Compute Hamming distances
    total_bits = sum(ht.size for ht in h_true)
    hd = sum(np.sum((hh - ht) ** 2) for hh, ht in zip(h, h_true)) / total_bits

    if return_layerwise:
        hd0 = np.sum((h[0] - h_true[0]) ** 2) / h_true[0].size
        hd1 = np.sum((h[1] - h_true[1]) ** 2) / h_true[1].size
        return hd, hd0, hd1, h

    if return_masks:
        return hd, h
    return hd


# ── SUB-EXPERIMENT A: MLP Phase Transition ────────────────────────────────

def run_subexp_a():
    print("=" * 60)
    print("Sub-exp A: MLP Phase Transition")
    print("=" * 60)

    N, H = 10, 5
    SIGMA = 0.01
    ETA = 1e-4
    ALPHA = 1.0
    T_GLAUBER = 10
    N_SEEDS = 4
    # rho range calibrated to MLP energy scale: rho_c ~ 5e-5
    RHO_VALS = np.concatenate([[0], np.logspace(-6, -2, 12)])

    regimes = {
        'overdetermined': 60,     # M > N_total_params=55
        'critical': 10,           # M = N_in
        'underdetermined': 5,     # M < N_in
    }

    results = []
    layerwise_results = []

    for regime_name, M in regimes.items():
        print(f"\n--- {regime_name} (M={M}) ---")
        for ri, rho in enumerate(RHO_VALS):
            hds, hd0s, hd1s = [], [], []
            n_actives = []
            for seed in range(N_SEEDS):
                rng_data = np.random.default_rng(seed)
                X, y, w_true, h_true = generate_mlp_data(N, H, M, SIGMA, rng_data)
                hd, hd0, hd1, h_final = mlp_glauber(
                    X, y, h_true, ETA, rho, ALPHA,
                    T_GLAUBER, seed=seed + 1000, return_layerwise=True)
                hds.append(hd)
                hd0s.append(hd0)
                hd1s.append(hd1)
                n_actives.append(sum(np.sum(hh) for hh in h_final))

            row = dict(regime=regime_name, M=M, rho=rho,
                       hamming_mean=np.mean(hds), hamming_std=np.std(hds),
                       active_mean=np.mean(n_actives), active_std=np.std(n_actives))
            results.append(row)

            if regime_name == 'overdetermined':
                layerwise_results.append(dict(
                    rho=rho,
                    hamming_layer0_mean=np.mean(hd0s),
                    hamming_layer0_std=np.std(hd0s),
                    hamming_layer1_mean=np.mean(hd1s),
                    hamming_layer1_std=np.std(hd1s)))

            print(f"  rho={rho:.7f}  hamming={np.mean(hds):.4f} ± {np.std(hds):.4f}"
                  f"  active={np.mean(n_actives):.1f}"
                  + (f"  [L0={np.mean(hd0s):.3f} L1={np.mean(hd1s):.3f}]"
                     if regime_name == 'overdetermined' else ''),
                  flush=True)

    # Save phase transition results
    out = '/home/petty/pruning-research/results/mlp_phase_transition.csv'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['regime', 'M', 'rho',
                                          'hamming_mean', 'hamming_std',
                                          'active_mean', 'active_std'])
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {out}")

    # Save layerwise results
    out_lw = '/home/petty/pruning-research/results/mlp_layerwise_transition.csv'
    with open(out_lw, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['rho', 'hamming_layer0_mean',
                                          'hamming_layer0_std',
                                          'hamming_layer1_mean',
                                          'hamming_layer1_std'])
        w.writeheader()
        w.writerows(layerwise_results)
    print(f"Saved: {out_lw}")

    return results, layerwise_results


# ── SUB-EXPERIMENT B: Jaccard Support Overlap (UWSH) ─────────────────────

def jaccard(h1, h2):
    """Jaccard similarity between two binary mask vectors."""
    s1 = set(np.where(h1.ravel() > 0.5)[0])
    s2 = set(np.where(h2.ravel() > 0.5)[0])
    if len(s1 | s2) == 0:
        return 1.0  # both empty = identical
    return len(s1 & s2) / len(s1 | s2)


def run_subexp_b(rho_c_estimate):
    print("\n" + "=" * 60)
    print("Sub-exp B: Jaccard Support Overlap (UWSH)")
    print("=" * 60)

    N, H = 10, 5
    M = 5  # underdetermined
    SIGMA = 0.01
    ETA = 1e-4
    ALPHA = 1.0
    T_GLAUBER = 10
    N_SEEDS = 15

    rho_vals = [0.0, rho_c_estimate / 2, rho_c_estimate, 2 * rho_c_estimate]
    rho_labels = ['0', 'rho_c/2', 'rho_c', '2*rho_c']

    results = []
    for rho, label in zip(rho_vals, rho_labels):
        print(f"\n--- rho={rho:.7f} ({label}) ---")
        masks_all = []
        for seed in range(N_SEEDS):
            rng_data = np.random.default_rng(0)  # same data across seeds
            X, y, w_true, h_true = generate_mlp_data(N, H, M, SIGMA, rng_data)
            _, h_final = mlp_glauber(X, y, h_true, ETA, rho, ALPHA,
                                     T_GLAUBER, seed=seed + 2000,
                                     return_masks=True)
            mask_flat = np.concatenate([hh.ravel() for hh in h_final])
            masks_all.append(mask_flat)
            if (seed + 1) % 5 == 0:
                n_act = np.sum(mask_flat > 0.5)
                print(f"  seed {seed+1}/{N_SEEDS} done  active={n_act}",
                      flush=True)

        # Compute pairwise Jaccard
        jaccards = []
        for i, j in combinations(range(N_SEEDS), 2):
            jaccards.append(jaccard(masks_all[i], masks_all[j]))

        support_sizes = [np.sum(m > 0.5) for m in masks_all]

        row = dict(rho=rho, rho_label=label,
                   jaccard_mean=np.mean(jaccards),
                   jaccard_std=np.std(jaccards),
                   support_size_mean=np.mean(support_sizes),
                   support_size_std=np.std(support_sizes))
        results.append(row)
        print(f"  Jaccard={np.mean(jaccards):.4f} ± {np.std(jaccards):.4f}  "
              f"support={np.mean(support_sizes):.1f} ± {np.std(support_sizes):.1f}")

    out = '/home/petty/pruning-research/results/mlp_jaccard.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['rho', 'rho_label', 'jaccard_mean',
                                          'jaccard_std', 'support_size_mean',
                                          'support_size_std'])
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {out}")

    # UWSH verdict
    j_0 = results[0]['jaccard_mean']
    j_c = results[2]['jaccard_mean']
    delta = j_c - j_0
    if delta > 0.1:
        print(f"\n>>> UWSH SUPPORTED (Jaccard at rho_c - Jaccard at 0 = {delta:.3f} > 0.1)")
    else:
        print(f"\n>>> UWSH WEAK (Jaccard at rho_c - Jaccard at 0 = {delta:.3f} <= 0.1)")

    return results


# ── MAIN ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t0 = time.time()

    # A + C: Phase transition + layer-wise Hamming
    results_a, results_c = run_subexp_a()

    # Layer-wise summary
    print("\n" + "=" * 60)
    print("Sub-exp C: Layer-wise Hamming (from overdetermined regime)")
    print("=" * 60)
    for r in results_c:
        print(f"  rho={r['rho']:.7f}  L0={r['hamming_layer0_mean']:.4f}  "
              f"L1={r['hamming_layer1_mean']:.4f}")

    # Estimate rho_c from sub-exp A (overdetermined regime)
    # Find rho where active count transitions from ~55 to near 0
    over_rows = [r for r in results_a if r['regime'] == 'overdetermined']
    rho_c = 0.00005  # default based on calibration
    for i, r in enumerate(over_rows):
        if r['active_mean'] < 27 and r['rho'] > 0:  # less than true active count
            rho_c = r['rho']
            break
    print(f"\nEstimated rho_c (overdetermined, active<27): {rho_c:.7f}")

    # B: Jaccard support overlap
    results_b = run_subexp_b(rho_c)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print("Done: Exp 20 MLP phase transition + UWSH complete")
