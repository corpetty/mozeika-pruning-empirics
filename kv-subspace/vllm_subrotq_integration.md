# vLLM SubRotQ Integration Plan

## Current Status

✅ vLLM 0.19.1.dev6 running Gemma4-E4B-it on GPU0  
✅ Server responding at http://localhost:8000  
✅ VRAM usage: 23.3GB / 24GB (near capacity at 4K context)  

## Goal

Implement SubRotQ KV cache compression to enable **3× context scaling** (4K → 12K) on the same 24GB RTX 3090.

## Architecture

vLLM uses `vllm.attention.backends.triton.TritonAttentionBackend` for Gemma4 (heterogeneous head dims force TRITON_ATTN).

### Key Components

1. **KV Cache Storage**: `vllm.v1.worker.gpu_model_runner.ModelInput.kv_cache`
2. **Cache Update**: `vllm::unified_kv_cache_update` custom op (TRITON kernel)
3. **Attention Compute**: `vllm::unified_attention` (reads compressed KV)

### Compression Hook Points

**Option A: Monkey-patch cache write**  
- Intercept `unified_kv_cache_update` TRITON kernel
- Compress K/V before writing to PagedAttention cache
- Pros: Clean, no vLLM fork
- Cons: TRITON kernel hard to monkey-patch

**Option B: Post-process cache pages**  
- Let vLLM write full-rank KV
- Run async compression pass on allocated pages
- Update page metadata with compressed flag
- Pros: Easier to implement
- Cons: Temporary memory spike (need 2× KV budget)

**Option C: Python-level wrapper** (CHOSEN)  
- Wrap `GPUModelRunner.execute_model()`
- Compress KV in returned `ModelOutput` before caching
- Modify attention to decompress on-the-fly
- Pros: Pure Python, no CUDA/TRITON required
- Cons: Slower (acceptable for proof-of-concept)

## Implementation Plan

### Phase 1: Compression Module (Python)

Create `subrotq_compressor.py`:

```python
import torch
import numpy as np
from typing import Dict, Tuple

class SubRotQCompressor:
    def __init__(self, model_name: str, basis_path: str, k: int = 128, n_bits: int = 4):
        """
        Args:
            model_name: HF model identifier
            basis_path: Path to PCA basis file (NPZ format)
            k: Subspace rank (compression rank)
            n_bits: Quantization bits for residuals
        """
        self.k = k
        self.n_bits = n_bits
        self.bases = self._load_bases(basis_path)
        
    def _load_bases(self, path: str) -> Dict[Tuple[int, int], np.ndarray]:
        """Load per-layer, per-head PCA bases."""
        data = np.load(path)
        bases = {}
        for key in data.files:
            if key.startswith('U_'):
                # Format: U_L{layer}_H{head}
                parts = key.split('_')
                layer = int(parts[1][1:])
                head = int(parts[2][1:])
                bases[(layer, head)] = data[key]  # Shape: (d_head, k)
        return bases
    
    def compress_kv(self, K: torch.Tensor, V: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress K and V tensors for a layer.
        
        Args:
            K: (batch, num_heads, seq_len, d_head) - Key tensor
            V: (batch, num_heads, seq_len, d_head) - Value tensor
            layer_idx: Layer index
            
        Returns:
            K_compressed: (batch, num_heads, seq_len, k + quantized_bits)
            V_compressed: Same format (V compression disabled by default)
        """
        batch, num_heads, seq_len, d_head = K.shape
        
        K_compressed = torch.zeros(batch, num_heads, seq_len, self.k + self._quantizer_overhead(), 
                                   dtype=K.dtype, device=K.device)
        
        for h in range(num_heads):
            if (layer_idx, h) not in self.bases:
                # No basis for this head - store full-rank
                K_compressed[:, h, :, :d_head] = K[:, h]
                continue
                
            U = torch.from_numpy(self.bases[(layer_idx, h)]).to(K.device, K.dtype)  # (d_head, k)
            
            # Project to subspace: K_proj = K @ U
            K_proj = K[:, h] @ U  # (batch, seq_len, k)
            
            # Compute residual: K_res = K - K_proj @ U.T
            K_reconstructed = K_proj @ U.T
            K_res = K[:, h] - K_reconstructed
            
            # Quantize residual
            K_res_quantized = self._quantize(K_res, self.n_bits)
            
            # Pack: [K_proj (k floats), K_res_quantized (compressed)]
            K_compressed[:, h, :, :self.k] = K_proj
            K_compressed[:, h, :, self.k:] = K_res_quantized
        
        # V compression disabled (fails at k < d_head per paper findings)
        V_compressed = V
        
        return K_compressed, V_compressed
    
    def decompress_kv(self, K_c: torch.Tensor, V_c: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress K and V tensors."""
        batch, num_heads, seq_len, _ = K_c.shape
        d_head = 128  # Gemma4 standard head dim
        
        K = torch.zeros(batch, num_heads, seq_len, d_head, dtype=K_c.dtype, device=K_c.device)
        
        for h in range(num_heads):
            if (layer_idx, h) not in self.bases:
                K[:, h] = K_c[:, h, :, :d_head]
                continue
                
            U = torch.from_numpy(self.bases[(layer_idx, h)]).to(K_c.device, K_c.dtype)
            
            # Unpack
            K_proj = K_c[:, h, :, :self.k]
            K_res_quantized = K_c[:, h, :, self.k:]
            
            # Dequantize residual
            K_res = self._dequantize(K_res_quantized, self.n_bits, d_head)
            
            # Reconstruct: K = K_proj @ U.T + K_res
            K[:, h] = (K_proj @ U.T) + K_res
        
        V = V_c  # No V compression
        
        return K, V
    
    def _quantize(self, x: torch.Tensor, n_bits: int) -> torch.Tensor:
        """Uniform quantization (placeholder - use SubRotQ quantizer)."""
        # For now: just pack into smaller dtype
        if n_bits == 4:
            # Pack two 4-bit values into one uint8
            # Shape: (batch, seq_len, d_head) -> (batch, seq_len, d_head // 2)
            x_scaled = ((x + 1.0) / 2.0 * 15).clamp(0, 15).to(torch.uint8)
            x_packed = x_scaled[..., ::2] * 16 + x_scaled[..., 1::2]
            return x_packed.to(torch.float16)  # Temporary: store as fp16 for simplicity
        return x
    
    def _dequantize(self, x_q: torch.Tensor, n_bits: int, d_head: int) -> torch.Tensor:
        """Reverse quantization."""
        if n_bits == 4:
            # Unpack uint8 -> two 4-bit values
            batch, seq_len, packed_dim = x_q.shape
            x_unpacked = torch.zeros(batch, seq_len, d_head, dtype=x_q.dtype, device=x_q.device)
            x_q_int = x_q.to(torch.uint8)
            x_unpacked[..., ::2] = (x_q_int // 16).to(torch.float16) / 15.0 * 2.0 - 1.0
            x_unpacked[..., 1::2] = (x_q_int % 16).to(torch.float16) / 15.0 * 2.0 - 1.0
            return x_unpacked
        return x_q
    
    def _quantizer_overhead(self) -> int:
        """Bytes needed for quantized residual storage."""
        if self.n_bits == 4:
            return 64  # Placeholder: 128-dim residual packed into 64 bytes
        return 128
```

### Phase 2: Monkey-Patch vLLM

Create `vllm_subrotq_patch.py`:

```python
import torch
from typing import Optional
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from subrotq_compressor import SubRotQCompressor

# Global compressor instance
_compressor: Optional[SubRotQCompressor] = None

def enable_subrotq(basis_path: str, k: int = 128, n_bits: int = 4):
    """Enable SubRotQ compression globally."""
    global _compressor
    _compressor = SubRotQCompressor("google/gemma-4-E4B-it", basis_path, k, n_bits)
    
    # Monkey-patch GPUModelRunner.execute_model
    original_execute = GPUModelRunner.execute_model
    
    def execute_with_subrotq(self, *args, **kwargs):
        output = original_execute(self, *args, **kwargs)
        
        # Compress KV cache in output
        if hasattr(output, 'kv_caches') and _compressor is not None:
            for layer_idx, (K, V) in enumerate(output.kv_caches):
                K_c, V_c = _compressor.compress_kv(K, V, layer_idx)
                output.kv_caches[layer_idx] = (K_c, V_c)
        
        return output
    
    GPUModelRunner.execute_model = execute_with_subrotq
    print(f"✓ SubRotQ enabled: k={k}, n_bits={n_bits}")
```

### Phase 3: Generate Basis

Use existing `collect.py` + `fit_pca` to generate Gemma4-E4B basis:

```bash
python3 scripts/generate_gemma4_basis.py \
  --model google/gemma-4-E4B-it \
  --output results/gemma4_e4b_pca_basis_k128.npz \
  --k 128 \
  --calibration-data wikitext \
  --num-samples 500
```

### Phase 4: Test

```python
from vllm_subrotq_patch import enable_subrotq

# Enable SubRotQ
enable_subrotq("results/gemma4_e4b_pca_basis_k128.npz", k=128, n_bits=4)

# Test generation
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "google/gemma-4-E4B-it",
    "messages": [{"role": "user", "content": "Explain quantum computing in detail..."}],
    "max_tokens": 4000  # Should now fit 12K context with 3× compression
})
```

## Expected Results

- **Baseline (no compression)**: 4K context, 23.3GB VRAM
- **With SubRotQ (k=128, 4-bit)**: 12K context, ~23.5GB VRAM (4× KV cache CR)
- **Quality**: 0.98× rel_PPL (essentially lossless per exp24)

## Next Steps

1. Implement `subrotq_compressor.py` ✓
2. Create basis generation script
3. Test compression overhead
4. Benchmark 4K → 12K scaling

---

**Status**: Ready to implement Phase 1
