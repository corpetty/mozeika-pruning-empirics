"""
calibration.py — Collect KV statistics and fit PCA bases for compression.

The basis captures the principal subspace of K and V vectors observed during
a calibration forward pass. At inference time, vectors are projected into this
subspace, quantized, and reconstructed.
"""

from __future__ import annotations

import gc
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .arch import ModelInfo, find_attention_layers


@dataclass
class KVBasis:
    """
    Fitted PCA bases for K and V at every (layer, head).

    bases_k[(li, hi)] = (U: np.ndarray(d_head, k), mean: np.ndarray(d_head,))
    bases_v[(li, hi)] = same shape

    k is stored per-basis (may differ if adaptive policy is used).
    """
    info: ModelInfo
    k: int                        # nominal subspace dim
    bits: int                     # quantization bits
    bases_k: Dict = field(default_factory=dict)
    bases_v: Dict = field(default_factory=dict)
    calibration_tokens: int = 0

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[kvpatch] Basis saved → {path}")

    @staticmethod
    def load(path: Union[str, Path]) -> "KVBasis":
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        print(f"[kvpatch] Basis loaded ← {path} "
              f"(k={obj.k}, bits={obj.bits}, toks={obj.calibration_tokens})")
        return obj

    @property
    def compression_ratio(self) -> float:
        """Theoretical CR vs 16-bit uncompressed."""
        return (self.info.d_head * 16) / (self.k * self.bits)


def calibrate(
    model,
    tokenizer,
    *,
    k: int = 112,
    bits: int = 4,
    info: Optional[ModelInfo] = None,
    texts: Optional[List[str]] = None,
    n_tokens: int = 2048,
    device: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> KVBasis:
    """
    Run a calibration forward pass and fit PCA bases for every (layer, head).

    Args:
        model:        HuggingFace or AWQ model.
        tokenizer:    Matching tokenizer.
        k:            Subspace dimension (≤ d_head). 112 = safe default for d_head=128.
        bits:         Quantization bits (4 or 8).
        info:         Pre-detected ModelInfo (auto-detected if None).
        texts:        Calibration texts. Defaults to a built-in mixed-domain corpus.
        n_tokens:     Max tokens to use for calibration.
        device:       Target device. Auto-detected if None.
        save_path:    If given, serialise the basis to this path.
        verbose:      Print progress.

    Returns:
        KVBasis ready to pass to patch().
    """
    from .arch import detect_arch

    if info is None:
        info = detect_arch(model)
        if verbose:
            print(f"[kvpatch] Detected arch={info.arch} | "
                  f"layers={info.n_layers} | kv_heads={info.n_kv_heads} | "
                  f"d_head={info.d_head} | qk_norm={info.has_qk_norm}")

    if device is None:
        device = _infer_device(model)

    if texts is None:
        texts = _default_calibration_texts()

    calib_text = " ".join(texts)

    if verbose:
        print(f"[kvpatch] Calibrating: k={k}, bits={bits}, "
              f"device={device}, n_tokens={n_tokens}")

    raw_kvs = _collect_raw_kvs(model, tokenizer, calib_text, n_tokens,
                                device, info, verbose=verbose)

    if verbose:
        print(f"[kvpatch] Fitting PCA bases for "
              f"{len(raw_kvs)} (layer, head) pairs ...")

    bases_k, bases_v = _fit_bases(raw_kvs, k, info.d_head)
    del raw_kvs
    gc.collect()

    basis = KVBasis(
        info=info,
        k=k,
        bits=bits,
        bases_k=bases_k,
        bases_v=bases_v,
        calibration_tokens=n_tokens,
    )

    if verbose:
        cr = basis.compression_ratio
        print(f"[kvpatch] Calibration done. "
              f"Theoretical CR = {cr:.2f}x vs fp16 | "
              f"KV memory ÷ {cr:.1f}")

    if save_path is not None:
        basis.save(save_path)

    return basis


# ── Internal ──────────────────────────────────────────────────────────────────

def _collect_raw_kvs(model, tokenizer, text, n_tokens, device, info, verbose=True):
    """
    Forward-hook collection of raw K and V vectors.
    Returns {(li, hi): {'K': np.ndarray(T, d_head), 'V': np.ndarray(T, d_head)}}
    """
    kvs: Dict[Tuple[int, int], Dict[str, List]] = {}
    hooks = []
    attn_layers = find_attention_layers(model)

    n_kv  = info.n_kv_heads
    d     = info.d_head

    for layer_idx, attn in attn_layers:

        def make_k_hook(li):
            def _hook(module, inp, output):
                x = output.detach().cpu().float()
                b, s, _ = x.shape
                xr = x.reshape(b, s, n_kv, d)
                for h in range(n_kv):
                    key = (li, h)
                    kvs.setdefault(key, {'K': [], 'V': []})
                    kvs[key]['K'].append(xr[0, :, h, :].numpy())
            return _hook

        def make_v_hook(li):
            def _hook(module, inp, output):
                x = output.detach().cpu().float()
                b, s, _ = x.shape
                xr = x.reshape(b, s, n_kv, d)
                for h in range(n_kv):
                    key = (li, h)
                    kvs.setdefault(key, {'K': [], 'V': []})
                    kvs[key]['V'].append(xr[0, :, h, :].numpy())
            return _hook

        hooks.append(attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))
        hooks.append(attn.v_proj.register_forward_hook(make_v_hook(layer_idx)))

    if verbose:
        print(f"[kvpatch] Running calibration pass ({n_tokens} tokens) ...")

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=n_tokens)
    ids = enc["input_ids"].to(device)
    with torch.no_grad():
        model(ids)

    for h in hooks:
        h.remove()

    # Concatenate per key
    return {
        key: {
            'K': np.concatenate(d_['K'], axis=0),
            'V': np.concatenate(d_['V'], axis=0),
        }
        for key, d_ in kvs.items()
    }


def _fit_bases(raw_kvs, k, d_head):
    """
    Fit full-rank (d_head) PCA bases then keep top-k components.
    Storing full-rank U lets callers experiment with different k at runtime.
    """
    from compress import fit_pca

    bases_k, bases_v = {}, {}
    for (li, hi), d in raw_kvs.items():
        Uk, mk = fit_pca(d['K'], d_head)   # (d_head, d_head), (d_head,)
        Uv, mv = fit_pca(d['V'], d_head)
        bases_k[(li, hi)] = (Uk, mk)       # full-rank; patcher slices [:, :k]
        bases_v[(li, hi)] = (Uv, mv)
    return bases_k, bases_v


def _infer_device(model) -> str:
    try:
        p = next(model.parameters())
        return str(p.device)
    except StopIteration:
        return "cuda" if torch.cuda.is_available() else "cpu"


def _default_calibration_texts() -> List[str]:
    """Built-in mixed-domain corpus: science, code, history."""
    return [
        (
            "The mitochondria are membrane-bound organelles found in the cytoplasm of "
            "eukaryotic cells. They generate most of the cell's supply of adenosine "
            "triphosphate, used as a source of chemical energy. Mitochondria have their "
            "own DNA, known as mitochondrial DNA, which is separate from the nuclear DNA "
            "found in the cell nucleus. This organelle has its own ribosomes and can "
            "synthesize some of its own proteins. The number of mitochondria in a cell "
            "varies widely by organism and tissue type. Many cells have only a single "
            "mitochondrion, whereas others can contain several thousand mitochondria. "
            "The organelle is composed of compartments that carry out specialized "
            "functions. These compartments or regions include the outer membrane, the "
            "intermembrane space, the inner membrane, the cristae, and the matrix."
        ),
        (
            "In computer science, a binary search tree is a rooted binary tree data "
            "structure with the key of each internal node being greater than all the "
            "keys in the respective node's left subtree and less than the ones in its "
            "right subtree. The time complexity of operations on the binary search tree "
            "is linear with respect to the height of the tree. Binary search trees allow "
            "binary search for fast lookup, addition, and removal of data items. Since "
            "the nodes in a BST are laid out so that each comparison skips about half of "
            "the remaining tree, the lookup performance is proportional to that of binary "
            "logarithm. BSTs were devised in the 1960s for the problem of efficient "
            "storage of labeled data and are attributed to Conway Berners-Lee and David "
            "Wheeler."
        ),
        (
            "The French Revolution was a period of radical political and societal change "
            "in France that began with the Estates General of 1789 and ended with the "
            "formation of the French Consulate in November 1799. Many of its ideas are "
            "considered fundamental principles of liberal democracy, while phrases like "
            "liberté, égalité, fraternité reappeared in other revolts, such as the 1917 "
            "Russian Revolution, and inspired campaigns for the abolition of slavery and "
            "universal suffrage. The values and institutions of the Revolution dominate "
            "French politics to this day."
        ),
    ]
