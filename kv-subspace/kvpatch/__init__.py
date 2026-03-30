"""
kvpatch — KV subspace compression as a drop-in model patch.

Usage:
    from kvpatch import patch, calibrate, unpatch

    model, tokenizer = load_your_model(...)

    # One-shot (calibrates then patches)
    patch(model, tokenizer, k=112, bits=4)

    # Or explicit calibration (reuse basis across sessions)
    basis = calibrate(model, tokenizer, k=112, bits=4)
    patch(model, basis=basis)

    # Remove compression (restore original forward pass)
    unpatch(model)
"""

from .patcher import patch, unpatch
from .calibration import calibrate, KVBasis
from .utils import compression_ratio, memory_delta_gb

__all__ = ["patch", "unpatch", "calibrate", "KVBasis", "compression_ratio", "memory_delta_gb"]
__version__ = "0.1.0"
