"""
patcher.py — Install/remove KV compression hooks on a live model.

patch() is idempotent: calling it twice removes the first set of hooks.
unpatch() removes all hooks registered by kvpatch.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .arch import ModelInfo, detect_arch, find_attention_layers
from .calibration import KVBasis, calibrate


# ── Public API ────────────────────────────────────────────────────────────────

_HANDLE_ATTR = "_kvpatch_hooks"   # stashed on the model object


def patch(
    model,
    tokenizer=None,
    *,
    basis: Optional[KVBasis] = None,
    k: int = 112,
    bits: int = 4,
    compress_k: bool = True,
    compress_v: bool = False,        # V off by default (exp20 finding)
    k_K: Optional[int] = None,      # override k for K specifically
    k_V: Optional[int] = None,      # override k for V specifically
    n_tokens: int = 2048,
    verbose: bool = True,
) -> KVBasis:
    """
    Patch a model in-place with KV subspace compression.

    If `basis` is None, calibration is run automatically (requires `tokenizer`).
    Returns the KVBasis used (so you can save/reuse it).

    Args:
        model:       Any HuggingFace or AWQ causal LM.
        tokenizer:   Required when basis=None.
        basis:       Pre-fitted KVBasis (skip calibration).
        k:           Subspace dimension. 112 recommended for d_head=128.
        bits:        Quantization bits (4 or 8).
        compress_k:  Compress K vectors (default: True).
        compress_v:  Compress V vectors (default: False — V is hard, see RESULTS.md).
        k_K, k_V:    Per-tensor k overrides (override the global `k`).
        n_tokens:    Calibration tokens if calibrating from scratch.
        verbose:     Print progress and stats.

    Returns:
        KVBasis used for compression.
    """
    # Remove any existing patches first
    unpatch(model, silent=True)

    if basis is None:
        if tokenizer is None:
            raise ValueError("Either provide a `basis` or a `tokenizer` for calibration.")
        info = detect_arch(model)
        basis = calibrate(
            model, tokenizer,
            k=k, bits=bits, info=info,
            n_tokens=n_tokens, verbose=verbose,
        )

    k_k = k_K if k_K is not None else basis.k
    k_v = k_V if k_V is not None else basis.k

    hooks = _install_hooks(
        model, basis,
        k_K=k_k, bits_K=basis.bits,
        k_V=k_v, bits_V=basis.bits,
        compress_k=compress_k,
        compress_v=compress_v,
    )
    setattr(model, _HANDLE_ATTR, hooks)

    if verbose:
        info = basis.info
        active = []
        if compress_k:
            active.append(f"K(k={k_k}/{basis.bits}bit, CR={_cr(k_k,basis.bits,info.d_head):.1f}x)")
        if compress_v:
            active.append(f"V(k={k_v}/{basis.bits}bit, CR={_cr(k_v,basis.bits,info.d_head):.1f}x)")
        print(f"[kvpatch] ✓ Patched {info.arch} | "
              f"layers={info.n_layers} | "
              f"hooks={len(hooks)} | "
              f"compressing: {', '.join(active) or 'nothing (dry run?)'}")
        mem = memory_delta_gb(basis, compress_k=compress_k, compress_v=compress_v)
        print(f"[kvpatch]   KV cache memory reduction: ~{abs(mem):.1f} GB per 32K-token context")

    return basis


def unpatch(model, *, silent: bool = False):
    """Remove all kvpatch hooks from the model."""
    hooks: List = getattr(model, _HANDLE_ATTR, [])
    for h in hooks:
        h.remove()
    if hooks:
        if hasattr(model, _HANDLE_ATTR):
            delattr(model, _HANDLE_ATTR)
        if not silent:
            print(f"[kvpatch] Removed {len(hooks)} hooks. Model restored to baseline.")
    elif not silent:
        print("[kvpatch] No active hooks found.")
    gc.collect()


# ── Hook installation ─────────────────────────────────────────────────────────

def _install_hooks(model, basis: KVBasis,
                   k_K: int, bits_K: int,
                   k_V: int, bits_V: int,
                   compress_k: bool, compress_v: bool) -> List:
    from compress import subspace_polar_quantize

    hooks = []
    info  = basis.info
    n_kv  = info.n_kv_heads
    d     = info.d_head

    attn_layers = find_attention_layers(model)

    for layer_idx, attn in attn_layers:

        if compress_k and layer_idx in {l for l, _ in basis.bases_k.keys() if True}:

            def make_k_hook(li, _k=k_K, _bits=bits_K):
                def _hook(module, inp, output):
                    dev, dty = output.device, output.dtype
                    x  = output.detach().cpu().float()
                    b, s, tot = x.shape
                    xr = x.reshape(b, s, n_kv, d)
                    for h in range(n_kv):
                        key = (li, h)
                        if key not in basis.bases_k:
                            continue
                        U_full, mean = basis.bases_k[key]
                        xh = xr[0, :, h, :].numpy()   # (T, d)
                        U  = U_full[:, :_k]             # (d, k)
                        xr[0, :, h, :] = torch.from_numpy(
                            subspace_polar_quantize(xh, _k, _bits, U, mean)
                        )
                    return xr.reshape(b, s, tot).to(dty).to(dev)
                return _hook

            hooks.append(attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))

        if compress_v and layer_idx in {l for l, _ in basis.bases_v.keys() if True}:

            def make_v_hook(li, _k=k_V, _bits=bits_V):
                def _hook(module, inp, output):
                    dev, dty = output.device, output.dtype
                    x  = output.detach().cpu().float()
                    b, s, tot = x.shape
                    xr = x.reshape(b, s, n_kv, d)
                    for h in range(n_kv):
                        key = (li, h)
                        if key not in basis.bases_v:
                            continue
                        U_full, mean = basis.bases_v[key]
                        xh = xr[0, :, h, :].numpy()
                        U  = U_full[:, :_k]
                        xr[0, :, h, :] = torch.from_numpy(
                            subspace_polar_quantize(xh, _k, _bits, U, mean)
                        )
                    return xr.reshape(b, s, tot).to(dty).to(dev)
                return _hook

            hooks.append(attn.v_proj.register_forward_hook(make_v_hook(layer_idx)))

    return hooks


# ── Utilities (also exported from __init__) ───────────────────────────────────

def _cr(k, bits, d, fp_bits=16):
    return (d * fp_bits) / (k * bits)


def compression_ratio(k: int, bits: int, d_head: int = 128, fp_bits: int = 16) -> float:
    """Theoretical compression ratio vs fp16 full-dim."""
    return _cr(k, bits, d_head, fp_bits)


def memory_delta_gb(basis: KVBasis,
                    ctx_len: int = 32768,
                    batch_size: int = 1,
                    compress_k: bool = True,
                    compress_v: bool = False) -> float:
    """
    Estimate KV cache memory saved (negative = savings) in GB.

    Full KV cache size = 2 × n_layers × n_kv_heads × ctx × d_head × 2 bytes (fp16)
    Compressed size    = layers × kv_heads × ctx × k × (bits/8) bytes
    """
    info     = basis.info
    fp16_per = info.n_layers * info.n_kv_heads * ctx_len * info.d_head * 2  # bytes per K or V
    full_kv  = 2 * fp16_per * batch_size

    comp_k = fp16_per  # default: uncompressed
    comp_v = fp16_per

    if compress_k:
        comp_k = info.n_layers * info.n_kv_heads * ctx_len * basis.k * (basis.bits // 8) * batch_size
    if compress_v:
        comp_v = info.n_layers * info.n_kv_heads * ctx_len * basis.k * (basis.bits // 8) * batch_size

    compressed_kv = comp_k + comp_v
    return (compressed_kv - full_kv) / (1024 ** 3)   # GB (negative = savings)
