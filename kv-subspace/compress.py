"""
compress.py — PolarQuant + QJL implementation, attention score distortion measurement.

Tests the hypothesis: compressing KV vectors in their principal subspace requires
fewer bits to preserve attention score quality than full-dimensional compression.
"""

import numpy as np


# ── PolarQuant ────────────────────────────────────────────────────────────────

def random_rotation_matrix(d: int, seed: int = 0) -> np.ndarray:
    """Generate a random orthogonal rotation matrix via QR decomposition."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    return Q.astype(np.float32)


def polar_quantize(x: np.ndarray, n_bits: int, R: np.ndarray = None) -> np.ndarray:
    """
    PolarQuant: random rotation + scalar quantization of each component.
    
    x: (N, d) float vectors
    n_bits: bits per scalar (e.g. 4, 8)
    R: optional pre-computed rotation matrix (d, d)
    
    Returns reconstructed vectors (N, d).
    """
    N, d = x.shape
    if R is None:
        R = random_rotation_matrix(d)

    # Rotate
    xr = x @ R.T  # (N, d)

    # Quantize each dimension independently (uniform)
    x_q = quantize_uniform(xr, n_bits)

    # Rotate back
    return x_q @ R


def quantize_uniform(x: np.ndarray, n_bits: int) -> np.ndarray:
    """Uniform scalar quantization to n_bits per value."""
    n_levels = 2 ** n_bits
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    scale = (x_max - x_min) / (n_levels - 1)
    scale = np.where(scale == 0, 1.0, scale)  # avoid div by zero

    # Quantize
    x_int = np.round((x - x_min) / scale).astype(np.int32)
    x_int = np.clip(x_int, 0, n_levels - 1)

    # Dequantize
    return x_int * scale + x_min


# ── QJL (1-bit Johnson-Lindenstrauss correction) ──────────────────────────────

def qjl_encode(x: np.ndarray, n_proj: int, seed: int = 42) -> np.ndarray:
    """
    QJL: project x onto n_proj random directions, return sign bits.
    x: (N, d)
    Returns: (N, n_proj) int8 array of {-1, +1}
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((x.shape[1], n_proj)).astype(np.float32)
    return np.sign(x @ G).astype(np.int8)


def qjl_estimate_dot(q: np.ndarray, k_signs: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Estimate q·k using QJL-encoded k.
    q: (d,) query vector (full precision)
    k_signs: (N, n_proj) sign bits
    G: (d, n_proj) random projection matrix
    Returns: (N,) estimated dot products
    """
    # E[sign(k·g_j) * (q·g_j)] ≈ (2/π) * ||k|| * (q·k/||q||·||k||) * ||q|| ... 
    # Simplified estimator: dot(q, G) * k_signs^T, averaged
    q_proj = q @ G  # (n_proj,)
    return (k_signs @ q_proj) * (np.pi / (2 * k_signs.shape[1]))


# ── Subspace compression ──────────────────────────────────────────────────────

def fit_pca(X: np.ndarray, k: int):
    """
    Fit PCA, return (U_k, mean) where U_k is (d, k) matrix of top-k components.
    X: (N, d)
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    U_k = Vt[:k].T  # (d, k)
    return U_k, mean


def subspace_polar_quantize(x: np.ndarray, k: int, n_bits: int, 
                             U_k: np.ndarray = None, mean: np.ndarray = None,
                             R: np.ndarray = None) -> np.ndarray:
    """
    Project to k-dim subspace, PolarQuant, reconstruct to d-dim.
    
    x: (N, d)
    k: subspace dimension
    n_bits: bits per scalar in the subspace
    U_k: (d, k) — if None, computed from x
    mean: (d,) — if None, computed from x
    R: (k, k) rotation matrix — if None, generated
    
    Returns reconstructed vectors (N, d).
    """
    if U_k is None or mean is None:
        U_k, mean = fit_pca(x, k)

    # Project to subspace
    xc = x - mean
    x_proj = xc @ U_k  # (N, k)

    # PolarQuant in k-dim space
    if R is None:
        R = random_rotation_matrix(k)
    x_proj_q = polar_quantize(x_proj, n_bits, R)

    # Reconstruct
    x_recon = x_proj_q @ U_k.T + mean  # (N, d)
    return x_recon


# ── Distortion metrics ────────────────────────────────────────────────────────

def attention_score_distortion(Q: np.ndarray, K_true: np.ndarray, K_compressed: np.ndarray,
                                scale: float = None) -> dict:
    """
    Measure how much compression distorts attention scores.
    
    Q: (T_q, d) query vectors (full precision)
    K_true: (T_k, d) original key vectors
    K_compressed: (T_k, d) compressed+decompressed key vectors
    
    Returns dict with multiple distortion measures.
    """
    d = Q.shape[1]
    if scale is None:
        scale = 1.0 / np.sqrt(d)

    # Raw dot products
    logits_true = Q @ K_true.T * scale      # (T_q, T_k)
    logits_comp = Q @ K_compressed.T * scale

    # Softmax attention weights
    def softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        ex = np.exp(x)
        return ex / ex.sum(axis=-1, keepdims=True)

    attn_true = softmax(logits_true)   # (T_q, T_k)
    attn_comp = softmax(logits_comp)

    # Metrics
    mse_logits = float(np.mean((logits_true - logits_comp) ** 2))
    mae_logits = float(np.mean(np.abs(logits_true - logits_comp)))

    # KL divergence: KL(p_true || p_comp)
    eps = 1e-10
    kl = float(np.mean(np.sum(attn_true * np.log((attn_true + eps) / (attn_comp + eps)), axis=-1)))

    # L1 distance between attention distributions
    l1_attn = float(np.mean(np.sum(np.abs(attn_true - attn_comp), axis=-1)))

    # Top-1 agreement: do they agree on the most attended token?
    top1_true = np.argmax(attn_true, axis=-1)
    top1_comp = np.argmax(attn_comp, axis=-1)
    top1_agreement = float(np.mean(top1_true == top1_comp))

    return {
        'mse_logits': mse_logits,
        'mae_logits': mae_logits,
        'kl_divergence': kl,
        'l1_attention': l1_attn,
        'top1_agreement': top1_agreement,
    }


# ── Main comparison experiment ────────────────────────────────────────────────

def compare_compression_methods(K: np.ndarray, V: np.ndarray, Q: np.ndarray,
                                  bit_budgets: list, k_values: list) -> list:
    """
    For a single head's K, V, Q vectors, compare:
      - Full-dim PolarQuant at various bit rates
      - Subspace PolarQuant at various (k, bits) with matched total bits/vector

    K, V, Q: (T, d_head) — single head vectors
    bit_budgets: total bits per vector to test (e.g. [2, 4, 8, 16, 32])
    k_values: subspace dimensions to test (e.g. [8, 16, 32, 64])

    Returns list of result dicts.
    """
    T, d = K.shape
    results = []

    # Fit PCA on K (use first half as "calibration", second half as "test")
    T_cal = T // 2
    U_k_full, mean_K = fit_pca(K[:T_cal], d)  # fit on calibration
    K_test = K[T_cal:]
    Q_test = Q[T_cal:] if Q.shape[0] > T_cal else Q

    for bits_total in bit_budgets:
        # ── Full-dim PolarQuant ──
        # bits_total bits per scalar, d scalars per vector
        # effective: d * bits_total bits per vector → bits_per_scalar = bits_total
        # (here bits_total represents bits-per-scalar for fair comparison)
        n_bits = bits_total
        R_full = random_rotation_matrix(d, seed=0)
        K_full_compressed = polar_quantize(K_test, n_bits, R_full)

        dist_full = attention_score_distortion(Q_test, K_test, K_full_compressed)
        results.append({
            'method': 'full_dim',
            'bits_per_scalar': n_bits,
            'bits_per_vector': n_bits * d,
            'k': d,
            **{f'K_{kk}': vv for kk, vv in dist_full.items()},
        })

        # ── Subspace PolarQuant at matched bits/vector ──
        for k in k_values:
            if k >= d:
                continue
            # Total bits per vector = k * bits_per_scalar_in_subspace
            # Match to full-dim: d * n_bits = k * n_bits_sub
            # → n_bits_sub = d * n_bits / k (may not be integer)
            # Round to nearest int, skip if < 1
            n_bits_sub = max(1, round(d * n_bits / k))

            # Fit subspace on calibration
            U_k, mean_K_sub = fit_pca(K[:T_cal], k)
            R_sub = random_rotation_matrix(k, seed=0)
            K_sub_compressed = subspace_polar_quantize(
                K_test, k, n_bits_sub, U_k, mean_K_sub, R_sub
            )

            dist_sub = attention_score_distortion(Q_test, K_test, K_sub_compressed)
            results.append({
                'method': 'subspace',
                'bits_per_scalar': n_bits_sub,
                'bits_per_vector': n_bits_sub * k,
                'k': k,
                **{f'K_{kk}': vv for kk, vv in dist_sub.items()},
            })

    return results


# ── High-level compression helper ────────────────────────────────────────────

def compress_vec(x_np: np.ndarray, method: str, k: int, n_bits: int,
                 U=None, mean=None) -> np.ndarray:
    """
    Compress a (T, d) array of KV vectors using the specified method.

    method : 'subspace' — project to k-dim PCA subspace then polar-quantize
             'full_dim' — polar-quantize in full dimension (no PCA)
             None       — no compression, return x_np unchanged
    """
    if method == 'subspace':
        return subspace_polar_quantize(x_np, k, n_bits, U, mean)
    elif method == 'full_dim':
        return polar_quantize(x_np, n_bits)
    return x_np
