"""
Shared helpers for UWSH subspace analysis (Experiment 19).
Extracted so tests can import without running the full experiment.
"""

import numpy as np


# ── Inline Glauber helpers (same as exp 16) ──────────────────────────────

def energy(w, h, X, y, eta, rho, alpha=1.0):
    pred = X @ (w * h)
    L = 0.5 * np.mean((pred - y) ** 2)
    reg = 0.5 * eta * np.sum(w ** 2)
    V = alpha * np.sum(h**2 * (h - 1)**2) + 0.5 * rho * np.sum(h)
    return L + reg + V


def optimize_w_adam(w, h, X, y, eta, K=30, lr=0.01):
    """K steps of Adam on loss w.r.t. w (h fixed)."""
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


def run_glauber_single(X, y, eta, rho, alpha, T, seed):
    """
    Single-replica zero-temperature Glauber dynamics.
    Returns (w_final, h_final).
    """
    N = X.shape[1]
    rng = np.random.default_rng(seed)

    h = np.ones(N, dtype=float)
    w = rng.normal(0, 0.1, N)

    # Initial weight optimization
    w = optimize_w_adam(w, h, X, y, eta, K=50)

    for t in range(T):
        order = rng.permutation(N)
        for j in order:
            h_try = h.copy()
            h_try[j] = 1.0 - h_try[j]

            w_try = optimize_w_adam(w, h_try, X, y, eta, K=15)

            E_curr = energy(w, h, X, y, eta, rho, alpha)
            E_try = energy(w_try, h_try, X, y, eta, rho, alpha)

            if E_try < E_curr:
                h = h_try
                w = w_try

        # Full re-optimization after each sweep
        w = optimize_w_adam(w, h, X, y, eta, K=30)

    return w, h


# ── Spectral analysis helpers ────────────────────────────────────────────

def participation_ratio(eigenvalues):
    """PR = (sum λ)² / sum(λ²). Low PR = concentrated in few directions."""
    eigenvalues = np.abs(eigenvalues)
    s = np.sum(eigenvalues)
    s2 = np.sum(eigenvalues ** 2)
    if s2 < 1e-30:
        return 0.0
    return s ** 2 / s2


def top_k_variance_fraction(singular_values, k=5):
    """Fraction of total variance in top-k singular values."""
    var = singular_values ** 2
    total = np.sum(var)
    if total < 1e-30:
        return 0.0
    return np.sum(var[:k]) / total


def mean_pairwise_cosine_similarity(W):
    """
    Average |cos θ| over all pairs of rows in W.
    Each row is a pruned weight vector.
    """
    n_runs = W.shape[0]
    if n_runs < 2:
        return 0.0

    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    W_normed = W / norms

    cos_mat = np.abs(W_normed @ W_normed.T)

    mask = ~np.eye(n_runs, dtype=bool)
    return np.mean(cos_mat[mask])


def spectral_analysis(W, k=5):
    """
    Full spectral analysis of a (n_runs, N) matrix of pruned weight vectors.
    Returns dict of metrics.
    """
    n_runs, N = W.shape

    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    gram = W @ W.T
    eigvals = np.linalg.eigvalsh(gram)
    eigvals = eigvals[::-1]  # descending

    pr = participation_ratio(eigvals)
    top_k_var = top_k_variance_fraction(S, k=k)
    cos_sim = mean_pairwise_cosine_similarity(W)

    return {
        'participation_ratio': pr,
        'top5_variance_frac': top_k_var,
        'mean_pairwise_cos_sim': cos_sim,
        'singular_values': S[:10],
        'eigenvalues': eigvals[:10],
    }
