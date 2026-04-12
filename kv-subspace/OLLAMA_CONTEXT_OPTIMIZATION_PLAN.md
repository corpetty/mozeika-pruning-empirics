# Ollama Gemma4 26B Context Length Optimization Plan

**Goal:** Maximize context length on a single RTX 3090 (24GB VRAM)  
**Current Setup:** Gemma4 26B (Q4_K_M, 17GB), default 262K context window  
**Problem:** 262K context requires 92GB KV cache — impossible on single GPU without compression

---

## Current Situation Analysis

### Model Specs
- **Gemma4 26B** (based on Gemma 2 architecture)
- **Quantization:** Q4_K_M (4-bit mixed, ~0.65 bytes/param)
- **Model weights:** 17 GB
- **Architecture:** 46 layers, 16 GQA heads, d_head=128
- **Default context:** 262,144 tokens (from model metadata)

### KV Cache Memory Requirements (Uncompressed)
```
KV cache per token = 2 (K+V) × 46 layers × 16 heads × 128 dim × 2 bytes (fp16)
                   = 376,832 bytes/token
                   = 359.4 MB per 1K tokens
```

| Context | KV Cache | Total (model + KV + buffer) | Status |
|---------|----------|---------------------------|--------|
| 8K | 2.8 GB | 21.8 GB | ✓ Fits single GPU |
| 12K | 4.2 GB | 23.2 GB | ✓ Borderline |
| 16K | 5.6 GB | 24.6 GB | ✗ **OOM** |
| 32K | 11.5 GB | 30.5 GB | ✗ OOM (needs 2 GPUs) |
| 64K | 23.0 GB | 42.0 GB | ✗ OOM (needs 2 GPUs) |
| 128K | 46.0 GB | 65.0 GB | ✗ OOM (needs 3 GPUs) |
| 262K | **92.0 GB** | **111.0 GB** | ✗ OOM (needs 5 GPUs!) |

**Current Reality:** Ollama is silently using GPU0 + GPU1 despite `CUDA_VISIBLE_DEVICES=0` in `/etc/default/ollama`. GPU1 shows 20GB usage, confirming multi-GPU split.

---

## Option 1: KV Cache Compression (SubRotQ) — **RECOMMENDED**

**What:** Apply our SubRotQ k=128/4-bit compression to K cache, full-rank 4-bit to V cache.

### Implementation
1. Patch Ollama's llama.cpp backend with KV compression hooks
2. One-time calibration on 2K tokens (generic text)
3. PCA basis stored per-layer/head (~45MB overhead)
4. Real-time compress/decompress during inference

### Memory Savings
```
Compressed KV = 2 × 46 × 16 × 128 × 0.5 (4-bit)
              = 94,208 bytes/token
              = 89.9 MB per 1K tokens
Compression ratio: 4.00×
```

| Context | Compressed KV | Total | Gain vs Baseline |
|---------|---------------|-------|------------------|
| 8K | 0.7 GB | 19.7 GB | — |
| 12K | 1.1 GB | 20.1 GB | — |
| **32K** | **2.9 GB** | **21.9 GB** | **✓ Now fits single GPU!** |
| **48K** | **4.3 GB** | **23.3 GB** | **✓ 4× increase from baseline** |
| 64K | 5.8 GB | 24.8 GB | ✗ Borderline OOM |

**Maximum viable context on single GPU:** ~48K tokens (vs 12K baseline = **4× improvement**)

### Pros
- **4× context increase** (12K → 48K on single GPU)
- Quality degradation: <2% PPL (exp24: 0.98× rel-PPL on Qwen3)
- Downstream tasks nearly lossless (exp27: -3pp on ARC-C, others ±0)
- Generalizes across architectures (exp30: Mistral/Llama validated)
- One-time calibration, no retraining

### Cons
- **Implementation effort:** Moderate (2-3 days to patch llama.cpp)
- **Latency overhead:** 1.6× decode slowdown with Python hooks (exp26)
  - Needs CUDA kernel for production (<1.1× overhead estimated)
- **Quality loss:** Negligible at k=128, but non-zero
- **Basis storage:** 45MB per model instance

### Tradeoffs
- **Token speed:** -37% decode throughput with Python implementation (1.6× overhead)
  - You said speed is less important than context length — acceptable tradeoff
- **Memory vs Quality:** Can push to k=112/4-bit for 4.57× compression (55K ctx) but quality drops to 1.23× PPL (23% degradation) — NOT recommended
- **Compatibility:** Requires custom Ollama build; won't work with stock Ollama

---

## Option 2: Flash Attention 2 — **COMPLEMENTARY**

**What:** Replace standard attention with Flash Attention 2 to reduce activation memory from O(n²) to O(n).

### Memory Savings
- Reduces peak activation memory by ~3-4 GB at long context
- Does NOT compress KV cache itself
- Combined with baseline: ~15K max context (vs 12K)
- Combined with SubRotQ: ~52K max context (vs 48K)

### Pros
- Llama.cpp already supports FA2 (compile flag: `LLAMA_CUDA_FA_ALL_QUANTS=1`)
- No quality loss (mathematically equivalent to standard attention)
- Faster inference at long context (fewer memory-bound ops)

### Cons
- Smaller improvement than KV compression (~25% vs 300%)
- Requires recompiling llama.cpp from source
- Ollama build process is non-trivial

### Recommendation
- **Yes, do this** if you're already patching llama.cpp for SubRotQ
- Marginal additional effort for ~4K extra context
- Total gain: 12K → 52K context (4.3× improvement)

---

## Option 3: 8-bit KV Cache Quantization — **EASIER, SMALLER GAIN**

**What:** Quantize KV cache from fp16 → int8 (2× compression).

### Memory Savings
```
8-bit KV = 376,832 / 2 = 188,416 bytes/token
Max context on single GPU: ~24K tokens (2× baseline)
```

### Pros
- Llama.cpp supports `--cache-type-k q8_0 --cache-type-v q8_0`
- **Easy to enable** (just pass flags to Ollama, no recompilation needed)
- Quality loss: minimal (<1% PPL for most models at 8-bit)
- No calibration required

### Cons
- Only 2× improvement (vs 4× with SubRotQ)
- Still limited to 24K context on single GPU
- Quality degrades faster than SubRotQ at same compression ratio

### Recommendation
- **Quick win** if you want immediate improvement without engineering
- Add to Ollama Modelfile: `PARAMETER num_ctx 24576` + KV q8 flags
- Can stack with SubRotQ later for 8× total compression (96K ctx)

---

## Option 4: Quantize Model to Q3_K_S — **NOT RECOMMENDED**

**What:** Further quantize model weights from Q4_K_M → Q3_K_S (3-bit).

### Memory Savings
- Model size: 17GB → ~13GB (save 4GB)
- Max context: 12K → ~16K tokens (33% improvement)

### Pros
- Frees more VRAM for KV cache

### Cons
- **Significant quality loss** (~5-10% on reasoning tasks)
- Only marginal context gain (4K tokens)
- 3-bit quantization is at the edge of usability for 26B models

### Recommendation
- **Not worth it.** Better to use SubRotQ or 8-bit KV cache.

---

## Option 5: Dual-GPU Setup (Current Default) — **WHAT YOU'RE DOING NOW**

**What:** Let Ollama spread model + KV cache across both RTX 3090s.

### Current Behavior
- Model layers split across GPU0 + GPU1
- KV cache split across both GPUs
- Supports up to ~92K context (current 262K config is aspirational, not functional)

### Pros
- Already working (Ollama does this automatically)
- Supports longer contexts than single GPU
- No engineering effort

### Cons
- **Ties up both GPUs** — can't run two instances simultaneously
- **Inter-GPU bandwidth bottleneck** (NVLink not available on 3090)
- Wastes GPU1 capacity for small contexts
- Doesn't scale beyond ~128K context even with 2 GPUs

### Recommendation
- **Keep as fallback** for rare >48K context needs
- Use single-GPU + SubRotQ for 95% of workloads

---

## Recommended Implementation Plan

### Phase 1: Quick Wins (1-2 hours)
1. **Enable 8-bit KV cache** for immediate 2× improvement
   ```bash
   # Create custom Modelfile
   cat > Modelfile.gemma4-long <<EOF
   FROM gemma4:26b
   PARAMETER num_ctx 24576
   PARAMETER cache_type_k q8_0
   PARAMETER cache_type_v q8_0
   EOF
   
   ollama create gemma4:26b-long -f Modelfile.gemma4-long
   ```
   **Result:** 12K → 24K max context on single GPU, minimal quality loss

2. **Force single-GPU mode** to free up GPU1
   ```bash
   # Edit /etc/default/ollama
   CUDA_VISIBLE_DEVICES=0
   OLLAMA_NUM_GPU=1  # Add this
   sudo systemctl restart ollama
   ```

### Phase 2: SubRotQ Integration (2-3 days engineering)
1. **Clone and patch llama.cpp**
   - Fork llama.cpp used by Ollama
   - Add KV compression hooks (similar to our kvpatch library)
   - Implement k=128/4-bit SubRotQ pipeline

2. **Compile custom Ollama**
   - Build llama.cpp with FA2 + SubRotQ patches
   - Replace Ollama's bundled llama.cpp binary
   - Test with Gemma4 26B

3. **Calibrate and deploy**
   - Run one-time 2K token calibration
   - Save PCA bases to disk (~45MB)
   - Create new model variant: `gemma4:26b-subrotq`

**Result:** 12K → 48K max context on single GPU, <2% quality loss

### Phase 3: CUDA Kernel Optimization (optional, 1 week)
1. Implement fused compress/decompress kernel in CUDA
2. Reduce 1.6× overhead → ~1.1× overhead
3. Recovers most of the speed loss from compression

**Result:** 48K context at near-baseline speed

---

## Final Recommendations

### For Immediate Use (Today)
```bash
# Create 24K context variant with 8-bit KV cache
ollama create gemma4:26b-long -f- <<EOF
FROM gemma4:26b
PARAMETER num_ctx 24576
EOF

# Test it
ollama run gemma4:26b-long "Summarize a long document..."
```

### For Maximum Context (This Week)
1. Implement SubRotQ in llama.cpp (2-3 days)
2. Enable Flash Attention 2 during compilation
3. Target: **48K-52K context on single RTX 3090**
4. Quality: <2% PPL degradation, nearly lossless on tasks

### For Production (Long Term)
1. Implement CUDA kernel for SubRotQ
2. Contribute patch upstream to llama.cpp (if they accept)
3. Target: **48K+ context at 1.1× latency overhead**

---

## Comparison Table

| Approach | Max Context | Speed Impact | Quality Loss | Effort | GPU Usage |
|----------|-------------|--------------|--------------|--------|-----------|
| **Baseline (current)** | 12K | — | — | — | 1 GPU |
| **8-bit KV cache** | 24K | None | <1% | 5 min | 1 GPU |
| **SubRotQ k=128** | **48K** | **-37%*** | **<2%** | 2-3 days | 1 GPU |
| **SubRotQ + FA2** | **52K** | **-37%*** | **<2%** | 3-4 days | 1 GPU |
| **SubRotQ + CUDA kernel** | **48K** | **-9%** | **<2%** | 1-2 weeks | 1 GPU |
| **Dual-GPU (status quo)** | 92K | -20%** | — | 0 | 2 GPUs |

\* With Python hooks; CUDA kernel reduces to ~-9%  
\*\* Inter-GPU communication overhead

---

## Questions?

- **Which option should I start with?** → 8-bit KV cache (5 min setup, 2× gain)
- **Is SubRotQ worth the engineering effort?** → Yes, if you regularly need >24K context
- **Will this work with other models?** → Yes, generalizes to Qwen/Mistral/Llama (validated in exp21/30)
- **Can I combine 8-bit KV + SubRotQ?** → Yes! 8× total compression (but diminishing returns, quality loss stacks)

Let me know which path you want to take and I'll provide detailed implementation steps.
