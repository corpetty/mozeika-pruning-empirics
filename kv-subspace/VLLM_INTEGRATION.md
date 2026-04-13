# SubRotQ + vLLM Integration Plan

## Why vLLM is Perfect

1. **Native Gemma4 support** — no GGUF conversion needed
2. **Python hooks** — easy to inject compression at attention layer
3. **Production-ready** — actual serving infrastructure
4. **PagedAttention** — KV cache is already managed in blocks, perfect for compression
5. **Active development** — vLLM team would likely merge this upstream

## Architecture

### vLLM KV Cache Flow
```
Input → Attention Layer → KV Cache (PagedAttention blocks)
                              ↓
                        [SubRotQ Hook]
                              ↓
                    Compressed K-cache (k=128, 4-bit)
```

### Implementation Points

1. **Calibration**: Generate PCA basis from HuggingFace Gemma4 (same as current pipeline)
2. **Compression Hook**: Inject into `vllm.attention.backends.flash_attn.FlashAttentionBackend`
3. **Decompression**: Before attention computation, decompress on-the-fly
4. **Memory Savings**: Reduce KV block size from 16KB → 4KB (4× compression)

## Implementation Steps

### Phase 1: Setup (15 min)
```bash
# Install vLLM
pip install vllm

# Verify Gemma4 loads
python3 -c "
from vllm import LLM
llm = LLM(model='google/gemma-2-27b-it', gpu_memory_utilization=0.9)
output = llm.generate('Hello!', max_tokens=10)
print(output[0].outputs[0].text)
"
```

### Phase 2: SubRotQ Compression Module (1-2 hours)
Create `vllm_subrotq.py`:
```python
import torch
import numpy as np

class SubRotQCompressor:
    def __init__(self, basis_file, rank=128, n_bits=4):
        # Load PCA basis from .subrotq file
        self.U, self.mean, self.scale = self.load_basis(basis_file)
        self.rank = rank
        self.n_bits = n_bits
        self.quantizer = self.build_quantizer(n_bits)
    
    def compress(self, k_cache):
        """
        k_cache: (batch, seq_len, n_heads, d_head)
        Returns: (batch, seq_len, n_heads, rank) quantized to n_bits
        """
        # Center
        k_centered = k_cache - self.mean
        
        # Project to k-dimensional subspace
        k_proj = k_centered @ self.U[:, :self.rank]  # (..., rank)
        
        # 4-bit quantization
        k_quant = self.quantizer.encode(k_proj)
        
        return k_quant
    
    def decompress(self, k_quant):
        """
        k_quant: compressed K-cache
        Returns: (batch, seq_len, n_heads, d_head) reconstructed
        """
        # Dequantize
        k_proj = self.quantizer.decode(k_quant)
        
        # Reconstruct via U^T
        k_recon = k_proj @ self.U[:, :self.rank].T + self.mean
        
        return k_recon
```

### Phase 3: Inject into vLLM Attention (2-3 hours)

Monkey-patch `vllm.model_executor.layers.attention.PagedAttention`:

```python
# Hook into vLLM's KV cache write
import vllm.model_executor.layers.attention as attn_module

original_write_kv = attn_module.PagedAttention.write_to_cache

def subrotq_write_kv(key, value, kv_cache):
    # Compress K before writing
    key_compressed = compressor.compress(key)
    
    # Store compressed K + full V (V compression fails per paper)
    original_write_kv(key_compressed, value, kv_cache)

# Replace
attn_module.PagedAttention.write_to_cache = subrotq_write_kv
```

### Phase 4: Test on Gemma4-26B (30 min)

```python
from vllm import LLM, SamplingParams

# Initialize with SubRotQ
llm = LLM(
    model="google/gemma-2-27b-it",
    gpu_memory_utilization=0.95,
    max_model_len=65536,  # 2× baseline context
    trust_remote_code=True
)

# Generate long-context response
prompts = ["Explain quantum mechanics in 2000 words:"]
outputs = llm.generate(prompts, SamplingParams(max_tokens=2000))

# Monitor VRAM
import torch
print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

## Expected Results

| Config | Context | VRAM | Compression |
|--------|---------|------|-------------|
| Baseline | 32K | ~22 GB | 1.0× |
| SubRotQ k=128 | 64K | ~23 GB | 4.0× KV |
| SubRotQ k=128 | 96K | ~23.5 GB | 4.0× KV |

**Target: 3× context scaling (32K → 96K) on single RTX 3090**

## Advantages Over llama.cpp/Ollama

1. ✅ **No GGUF conversion** — use HuggingFace model directly
2. ✅ **No multimodal incompatibility** — vLLM handles Gemma4 vision correctly
3. ✅ **Python integration** — easier to prototype and debug
4. ✅ **Production-ready** — this is what people actually use for serving
5. ✅ **Upstreamable** — vLLM team actively maintains KV cache optimizations

## Risks

- vLLM's PagedAttention might complicate compression (need to handle blocks)
- First attempt may need iteration to get hooks right
- vLLM updates frequently — monkey-patching may break

## Timeline

- **Phase 1**: 15 min (install + verify)
- **Phase 2**: 1-2 hours (SubRotQ module)
- **Phase 3**: 2-3 hours (vLLM integration)
- **Phase 4**: 30 min (testing + measurement)

**Total: 4-6 hours to working prototype**

## Deliverables

1. Working SubRotQ compression in vLLM
2. Gemma4-26B running at 64-96K context on single RTX 3090
3. VRAM usage measurements
4. Quality assessment (sample outputs)
5. **Publishable results** — "SubRotQ enables 3× context scaling in production inference"

---

**Decision: Proceed with vLLM integration?**

This is the cleanest path to your goal: Gemma4 + SubRotQ + 3× context scaling.
