# Task: Integrate SubRotQ into vLLM for Gemma4-26B Context Scaling

## Goal
Implement SubRotQ KV-cache compression in vLLM to achieve 3× context scaling (32K → 96K tokens) on Gemma4-26B running on a single RTX 3090 24GB GPU.

## Context
- **Machine**: bugger (Ubuntu 24.04, dual RTX 3090, 64 CPUs)
- **GPU allocation**: Use GPU0 for testing (CUDA_VISIBLE_DEVICES=0). GPU1 is reserved for Ollama.
- **Working directory**: /home/petty/pruning-research/kv-subspace/
- **Existing assets**:
  - PCA basis: `results/subrotq_basis_mistral7b_k128.bin` (17 MB, k=128 rank, 4-bit quantization)
  - Compression code: `compress.py` (SubRotQ implementation with `subspace_rotation_quantize`)
  - Binary format: 64-byte header + per-layer-head U/mean/scale matrices

## Implementation Plan

### Phase 1: Setup & Verification (15 min)
1. **vLLM already installed** at `/home/petty/vllm-srv/` (see TOOLS.md)
   - Start script: `bash /home/petty/vllm-srv/start.sh`
   - venv: `/home/petty/vllm-srv/venv`
   - Default: Qwen3-14B-AWQ on GPU1, port 8000
   - For this task: Use GPU0 explicitly: `--gpu 0`

2. Test Gemma baseline (using vLLM Python API, not server mode):
   ```python
   from vllm import LLM, SamplingParams
   import torch
   
   llm = LLM(
       model="google/gemma-2-27b-it",
       gpu_memory_utilization=0.9,
       max_model_len=32768,
       trust_remote_code=True,
       tensor_parallel_size=1
   )
   
   output = llm.generate(
       "Explain quantum entanglement:",
       SamplingParams(max_tokens=100)
   )
   
   print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
   print(output[0].outputs[0].text)
   ```
3. Record baseline VRAM usage at 32K context

### Phase 2: SubRotQ Compression Module (1-2 hours)
Create `vllm_subrotq.py` with:
- `SubRotQCompressor` class
- `load_basis(basis_file)` — parse .subrotq binary format (reuse logic from llama-subrotq-loader.cpp)
- `compress(k_cache)` — PCA projection + 4-bit quantization
- `decompress(k_quant)` — dequantize + reconstruct via U^T
- Use existing `subspace_rotation_quantize` from `compress.py` as reference

**Binary format** (from existing code):
```
Header (64 bytes):
  - magic (4 bytes): "SRTQ"
  - version (4 bytes): 1
  - n_layers (4 bytes)
  - n_heads (4 bytes)
  - rank (4 bytes)
  - d_head (4 bytes)
  - n_bits (4 bytes)
  - reserved (36 bytes)

Per layer per head:
  - U matrix: (d_head × rank) float32
  - mean vector: (d_head,) float32
  - scale vector: (rank,) float32
```

### Phase 3: vLLM Integration (2-3 hours)
Inject SubRotQ hooks into vLLM's KV cache layer:

**Option A: Monkey-patch PagedAttention**
```python
import vllm.model_executor.layers.attention as attn

compressor = SubRotQCompressor("results/subrotq_basis_mistral7b_k128.bin", rank=128, n_bits=4)

original_forward = attn.PagedAttention.forward

def subrotq_forward(self, query, key, value, kv_cache, ...):
    # Compress K before storing
    key_compressed = compressor.compress(key)
    
    # Decompress K before attention
    key_decompressed = compressor.decompress(key_compressed)
    
    return original_forward(self, query, key_decompressed, value, kv_cache, ...)

attn.PagedAttention.forward = subrotq_forward
```

**Option B: Custom attention backend** (if monkey-patching fails)
- Subclass `vllm.attention.backends.abstract.AttentionBackend`
- Override `forward()` with SubRotQ compress/decompress
- Register via `vllm.attention.selector.get_attn_backend()`

**Key considerations**:
- vLLM uses PagedAttention (block-based KV cache) — compression must handle blocks
- K/V caches are stored separately in `kv_cache.key_cache` and `kv_cache.value_cache`
- Only compress K (V compression fails per exp20/exp21 results)
- Ensure compression happens after K is computed but before storage
- Decompression must happen before attention QK^T matmul

### Phase 4: Testing & Measurement (30 min)
Test context scaling at multiple lengths:

```python
test_configs = [
    {"ctx": 16384, "desc": "Baseline 16K"},
    {"ctx": 32768, "desc": "Baseline 32K"},
    {"ctx": 49152, "desc": "SubRotQ 48K (1.5×)"},
    {"ctx": 65536, "desc": "SubRotQ 64K (2×)"},
    {"ctx": 98304, "desc": "SubRotQ 96K (3×)"},
]

for cfg in test_configs:
    torch.cuda.reset_peak_memory_stats()
    
    llm = LLM(
        model="google/gemma-2-27b-it",
        max_model_len=cfg["ctx"],
        gpu_memory_utilization=0.95,
        trust_remote_code=True
    )
    
    output = llm.generate(
        "Explain quantum mechanics in detail:" + " Continue." * 100,
        SamplingParams(max_tokens=500)
    )
    
    vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"{cfg['desc']}: {vram_gb:.2f} GB VRAM, {len(output[0].outputs[0].text)} chars")
```

Save results to `results/vllm_context_scaling.json`.

## Technical Notes

### GPU Allocation Rule
**Always use GPU0 for vLLM testing:**
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```
**Never touch GPU1** — it's allocated to Ollama service.

### Gemma Architecture
- Gemma-2-27B: 46 layers, 32 attention heads, 4096 embedding dim
- Gemma4-26B (if available): 30 layers, 16 attention heads, 2816 embedding dim
- Both have d_head=128 (compatible with k=128 SubRotQ)
- Use `google/gemma-2-27b-it` from HuggingFace (instruction-tuned variant)

### Calibration Data Mismatch
- Existing basis: Mistral-7B (32 layers, 8 KV heads per layer)
- Target model: Gemma-2-27B (46 layers, architecture mismatch)

**Options**:
1. Test cross-architecture transfer (Mistral → Gemma) — may degrade quality
2. Generate Gemma-specific basis first using `calibrate_subrotq_basis.py`

**Recommendation**: Try Mistral basis first (faster). If quality degrades, generate Gemma basis.

### Expected Challenges
1. **PagedAttention block alignment** — compression must preserve block structure
2. **Tensor shapes** — vLLM may use different (batch, seq, head, dim) layouts than our code
3. **Performance** — Python compression overhead vs CUDA kernels
4. **Memory allocation** — vLLM's memory manager may need adjustment for compressed cache

### Success Criteria
- ✅ Gemma4-26B loads and generates text
- ✅ SubRotQ compression activates without crashes
- ✅ Context scaling: 32K → 96K (3×) fits in 24GB VRAM
- ✅ Output quality: coherent text generation (qualitative check)
- ✅ VRAM savings: ~4× KV cache reduction measurable

### Deliverables
1. `vllm_subrotq.py` — compression module
2. `test_vllm_subrotq.py` — integration test script
3. `results/vllm_context_scaling.json` — VRAM measurements
4. Sample outputs at 64K/96K context (saved to `results/vllm_samples/`)

## Budget & Time
- **Time estimate**: 4-6 hours
- **Cost cap**: $5 (Claude Code Sonnet 4)
- **Output**: Working prototype demonstrating 3× context scaling on single GPU

## Constraints
- **No file deletion** — never use `rm`, `git rm`, or `trash`
- **GPU discipline** — CUDA_VISIBLE_DEVICES=0 always, never use GPU1
- **SLURM not required** — interactive testing is fine (vLLM already set up)
- **Venv**: `/home/petty/vllm-srv/venv` (use this for vLLM Python imports)
- **PyTorch venv**: `/home/petty/torch-env` if torch imports needed outside vLLM

## References
- vLLM docs: https://docs.vllm.ai/
- Existing SubRotQ implementation: `compress.py` (lines 180-220)
- PCA basis loader reference: `/tmp/llama.cpp/src/llama-subrotq-loader.cpp`
- Calibration script: `calibrate_subrotq_basis.py`

---

**Start with Phase 1: Install vLLM and verify Gemma-2-27B baseline loads on GPU0.**
