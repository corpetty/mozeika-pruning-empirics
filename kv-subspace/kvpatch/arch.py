"""
arch.py — Architecture detection and attention layer discovery.

Supports: Qwen2/Qwen3, LLaMA-2/3, Mistral, Phi-3, Falcon, and generic fallback.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Tuple, Any


@dataclass
class ModelInfo:
    """Detected architecture parameters."""
    arch: str               # e.g. "qwen3", "llama", "mistral", "phi3", "unknown"
    n_layers: int
    n_kv_heads: int
    d_head: int
    has_qk_norm: bool       # Qwen3 applies RMSNorm to k_proj/q_proj outputs
    model_body: Any         # reference to the transformer body module
    lm_head: Any            # reference to the LM head

    def __getstate__(self):
        """Exclude live model references from pickle — they can't serialize."""
        state = self.__dict__.copy()
        state["model_body"] = None
        state["lm_head"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.d_head


def detect_arch(model) -> ModelInfo:
    """
    Auto-detect architecture and return ModelInfo.
    Raises ValueError if we can't find k_proj/v_proj layers.
    """
    cfg = getattr(model, "config", None)
    arch = _detect_arch_name(model, cfg)

    n_kv_heads = _get_cfg(cfg, ["num_key_value_heads"], default=None)
    n_heads     = _get_cfg(cfg, ["num_attention_heads"], default=None)
    hidden_size = _get_cfg(cfg, ["hidden_size"], default=None)
    head_dim    = _get_cfg(cfg, ["head_dim"], default=None)

    if head_dim is None and n_heads and hidden_size:
        head_dim = hidden_size // n_heads
    if n_kv_heads is None:
        n_kv_heads = n_heads  # MHA fallback

    if head_dim is None or n_kv_heads is None:
        raise ValueError(
            f"Could not infer n_kv_heads/d_head from config. "
            f"Pass them explicitly to patch(). Got: {cfg}"
        )

    # Count layers
    n_layers = _count_layers(model, arch)

    # Model body and lm_head
    body, lm_head = _get_body_and_head(model, arch)

    # QK-norm: only Qwen3 adds RMSNorm to k/q proj outputs
    has_qk_norm = arch in ("qwen3",)

    return ModelInfo(
        arch=arch,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        d_head=head_dim,
        has_qk_norm=has_qk_norm,
        model_body=body,
        lm_head=lm_head,
    )


def find_attention_layers(model) -> List[Tuple[int, Any]]:
    """
    Return [(layer_idx, attn_module), ...] in order.
    attn_module is the object that has k_proj and v_proj attributes.
    """
    layers = []

    # Standard: model.model.layers[i].self_attn  (Qwen3, LLaMA, Mistral)
    body = getattr(model, "model", None)
    if body is not None:
        sublayers = getattr(body, "layers", None)
        if sublayers is not None:
            for i, layer in enumerate(sublayers):
                attn = getattr(layer, "self_attn", None)
                if attn is not None and hasattr(attn, "k_proj"):
                    layers.append((i, attn))
            if layers:
                return layers

    # Phi-3 variant: model.model.layers[i].self_attn but projected via qkv_proj
    # (handled by the generic fallback below)

    # Generic fallback: scan named_modules for anything with k_proj + v_proj
    seen = set()
    for name, module in model.named_modules():
        if hasattr(module, "k_proj") and hasattr(module, "v_proj"):
            if id(module) not in seen:
                # Extract layer index from name if possible
                nums = re.findall(r"\d+", name)
                idx = int(nums[-1]) if nums else len(layers)
                layers.append((idx, module))
                seen.add(id(module))

    if not layers:
        raise ValueError(
            "Could not find attention layers with k_proj/v_proj. "
            "Your architecture may need a custom adapter — see kvpatch/arch.py."
        )

    layers.sort(key=lambda x: x[0])
    return layers


# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_arch_name(model, cfg) -> str:
    model_type = getattr(cfg, "model_type", "") or ""
    cls_name   = type(model).__name__.lower()

    if "qwen3" in model_type or "qwen3" in cls_name:
        return "qwen3"
    if "qwen2" in model_type or "qwen2" in cls_name:
        return "qwen2"
    if "llama" in model_type or "llama" in cls_name:
        return "llama"
    if "mistral" in model_type or "mistral" in cls_name:
        return "mistral"
    if "phi3" in model_type or "phi-3" in model_type or "phi3" in cls_name:
        return "phi3"
    if "falcon" in model_type or "falcon" in cls_name:
        return "falcon"
    return "unknown"


def _get_cfg(cfg, keys, default=None):
    if cfg is None:
        return default
    for k in keys:
        v = getattr(cfg, k, None)
        if v is not None:
            return v
    return default


def _count_layers(model, arch) -> int:
    body = getattr(model, "model", None)
    if body is not None:
        layers = getattr(body, "layers", None)
        if layers is not None:
            return len(layers)
    # Fallback: count from find_attention_layers
    return len(find_attention_layers(model))


def _get_body_and_head(model, arch):
    """Return (transformer_body, lm_head) for chunked CE computation."""
    # AWQ models: model.model is the body, model.lm_head is the head
    # Standard HF: same layout
    body    = getattr(model, "model", model)
    lm_head = getattr(model, "lm_head", None)
    return body, lm_head
