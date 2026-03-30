"""
compress.py — KV cache compression implementations.

Two quantization backends, both operating on PCA-projected subspace vectors:

  SubRotQ (Subspace Rotation Quantization):
    Our method. Random orthogonal preconditioning (QR) + uniform scalar
    quantization per dimension. Fast, no calibration needed for the quantizer.

  PolarQuant (Han et al., arXiv:2502.02617):
    Random preconditioning + recursive polar coordinate transform + uniform
    quantization of polar angles. The key insight: after random preconditioning,
    polar angles concentrate around analytically known values, so no per-block
    normalization constants need to be stored.

Both are composed with PCA subspace projection in subspace_compress().
"""

import numpy as np


# ── Random orthogonal preconditioning (shared by both methods) ────────────────

def random_rotation_matrix(d: int, seed: int = 0) -> np.ndarray:
    """Random orthogonal matrix via QR decomposition (random preconditioning)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    return Q.astype(np.float32)


# ── SubRotQ: random rotation + uniform scalar quantization ───────────────────

def quantize_uniform(x: np.ndarray, n_bits: int) -> np.ndarray:
    """Uniform scalar quantization to n_bits per value, per-column scale."""
    n_levels = 2 ** n_bits
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    scale = (x_max - x_min) / (n_levels - 1)
    scale = np.where(scale == 0, 1.0, scale)
    x_int = np.clip(np.round((x - x_min) / scale).astype(np.int32), 0, n_levels - 1)
    return x_int * scale + x_min


def subrotq_quantize(x: np.ndarray, n_bits: int, R: np.ndarray = None) -> np.ndarray:
    """
    SubRotQ: random orthogonal preconditioning + uniform scalar quantization.

    This is our baseline method (previously mislabelled 'PolarQuant' in early
    experiments). Random rotation spreads variance across dimensions; uniform
    quantization then treats all dimensions identically.

    x: (N, d) float vectors
    n_bits: bits per scalar
    R: pre-computed (d, d) rotation matrix; generated if None
    Returns: reconstructed (N, d)
    """
    N, d = x.shape
    if R is None:
        R = random_rotation_matrix(d)
    xr = x @ R.T           # precondition
    xq = quantize_uniform(xr, n_bits)
    return xq @ R          # rotate back


# ── PolarQuant: random preconditioning + recursive polar transform ────────────

def _polar_to_cartesian(r: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Reconstruct Cartesian vectors from radius r and (d-1) polar angles.
    Inverse of the recursive polar transform.

    r: (N,) radius (L2 norm of preconditioned vector)
    angles: (N, d-1) — angles[i, j] in [-pi, pi] for j < d-2, else [-pi/2, pi/2]
    Returns: (N, d)
    """
    N, d_minus1 = angles.shape
    d = d_minus1 + 1
    # Build sin/cos products recursively
    x = np.zeros((N, d), dtype=np.float32)
    sin_prod = np.ones(N, dtype=np.float32)
    for i in range(d - 1):
        x[:, i] = sin_prod * np.cos(angles[:, i])
        sin_prod = sin_prod * np.sin(angles[:, i])
    x[:, d - 1] = sin_prod
    return x * r[:, None]


def polar_quantize_true(x: np.ndarray, n_bits: int, R: np.ndarray = None) -> np.ndarray:
    """
    True PolarQuant (Han et al., arXiv:2502.02617).

    Algorithm:
      1. Apply random orthogonal preconditioning R.
      2. Compute L2 radius r = ||Rx||.
      3. Normalise: x_norm = Rx / r  (unit sphere).
      4. Convert x_norm to (d-1) polar angles via recursive arctan2.
      5. After preconditioning, angles are distributed ~Uniform[-π,π] (first d-2)
         or ~Uniform[-π/2,π/2] (last angle). Quantise uniformly with fixed range
         [−π, π] or [−π/2, π/2] — no per-block scale/offset needed.
      6. Reconstruct unit vector from quantised angles, rescale by r.
      7. Invert preconditioning: R^T * x_recon.

    x: (N, d) float vectors
    n_bits: bits per scalar
    R: pre-computed (d, d) rotation; generated if None
    Returns: reconstructed (N, d)
    """
    N, d = x.shape
    if R is None:
        R = random_rotation_matrix(d)

    # 1. Precondition
    xr = (x @ R.T).astype(np.float64)  # (N, d)

    # 2. Radius
    r = np.linalg.norm(xr, axis=1)  # (N,)
    safe_r = np.where(r == 0, 1.0, r)

    # 3. Normalise to unit sphere
    x_norm = xr / safe_r[:, None]  # (N, d)

    # 4. Recursive polar angle extraction
    # angles[:, i] = arctan2(sin_prod_i, x_norm[:, i])
    # where sin_prod_i = product of sin(angles[:, j]) for j < i
    angles = np.zeros((N, d - 1), dtype=np.float64)
    sin_prod = np.ones(N, dtype=np.float64)
    for i in range(d - 1):
        if i < d - 2:
            # arctan2(remaining_norm, current_component) gives angle in [0, π]
            # but we keep signed via arctan2 to stay in [-π, π]
            remaining = np.linalg.norm(x_norm[:, i+1:], axis=1)
            angles[:, i] = np.arctan2(remaining, x_norm[:, i])
        else:
            # Last angle: arctan2(x[d-1], x[d-2]) gives full [-π, π]
            angles[:, i] = np.arctan2(x_norm[:, d-1], x_norm[:, d-2])
        sin_prod = sin_prod * np.sin(angles[:, i])

    # 5. Quantise angles with fixed range (no per-block scale needed after preconditioning)
    n_levels = 2 ** n_bits
    # First d-2 angles: range [0, π] (arctan2 of norm vs component)
    # Last angle: range [-π, π]
    angles_q = angles.copy()
    if d > 2:
        # angles 0..d-3: [0, π]
        step_mid = np.pi / (n_levels - 1)
        angles_q[:, :d-2] = np.round(angles[:, :d-2] / step_mid) * step_mid
        angles_q[:, :d-2] = np.clip(angles_q[:, :d-2], 0.0, np.pi)
    # Last angle: [-π, π]
    step_last = 2 * np.pi / (n_levels - 1)
    angles_q[:, d-2] = np.round((angles[:, d-2] + np.pi) / step_last) * step_last - np.pi
    angles_q[:, d-2] = np.clip(angles_q[:, d-2], -np.pi, np.pi)

    # 6. Reconstruct and rescale
    x_recon_norm = _polar_to_cartesian(np.ones(N, dtype=np.float32),
                                        angles_q.astype(np.float32))
    x_recon = x_recon_norm * safe_r.astype(np.float32)[:, None]

    # 7. Invert preconditioning
    return (x_recon @ R).astype(np.float32)


# ── Legacy alias (keeps old call sites working) ───────────────────────────────

def polar_quantize(x: np.ndarray, n_bits: int, R: np.ndarray = None) -> np.ndarray:
    """Legacy alias — calls SubRotQ (was mislabelled PolarQuant in early experiments)."""
    return subrotq_quantize(x, n_bits, R)


def quantize_uniform_legacy(x: np.ndarray, n_bits: int) -> np.ndarray:
    """Alias kept for backward compatibility."""
    return quantize_uniform(x, n_bits)


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


def subspace_compress(x: np.ndarray, k: int, n_bits: int,
                       U_k: np.ndarray = None, mean: np.ndarray = None,
                       R: np.ndarray = None,
                       quantizer: str = 'subrotq') -> np.ndarray:
    """
    Project to k-dim PCA subspace, quantize, reconstruct to d-dim.

    x: (N, d)
    k: subspace dimension
    n_bits: bits per scalar in the subspace
    U_k: (d, k) PCA basis — computed from x if None
    mean: (d,) PCA mean — computed from x if None
    R: (k, k) rotation matrix — generated if None
    quantizer: 'subrotq' (random rotation + uniform) or 'polarquant' (Han et al.)

    Returns reconstructed (N, d).
    """
    if U_k is None or mean is None:
        U_k, mean = fit_pca(x, k)

    # Project to subspace
    xc = x - mean
    x_proj = xc @ U_k  # (N, k)

    # Quantize in k-dim space
    if R is None:
        R = random_rotation_matrix(k)
    if quantizer == 'polarquant':
        x_proj_q = polar_quantize_true(x_proj, n_bits, R)
    else:  # default: subrotq
        x_proj_q = subrotq_quantize(x_proj, n_bits, R)

    # Reconstruct to d-dim
    return x_proj_q @ U_k.T + mean


def subspace_polar_quantize(x: np.ndarray, k: int, n_bits: int,
                             U_k: np.ndarray = None, mean: np.ndarray = None,
                             R: np.ndarray = None) -> np.ndarray:
    """Legacy alias — calls subspace_compress with SubRotQ backend."""
    return subspace_compress(x, k, n_bits, U_k, mean, R, quantizer='subrotq')


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
                 U=None, mean=None, quantizer: str = 'subrotq') -> np.ndarray:
    """
    Compress a (T, d) array of KV vectors.

    method    : 'subspace' — PCA projection then quantize
                'full_dim' — quantize in full dimension (no PCA)
                None       — no compression
    quantizer : 'subrotq' (default) or 'polarquant' — backend for quantization step
    """
    if method == 'subspace':
        return subspace_compress(x_np, k, n_bits, U, mean, quantizer=quantizer)
    elif method == 'full_dim':
        if quantizer == 'polarquant':
            return polar_quantize_true(x_np, n_bits)
        return subrotq_quantize(x_np, n_bits)
    return x_np
