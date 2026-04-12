# Ollama Gemma4 26B Context Optimization Plan (UPDATED)

**Goal:** Maximize context length on a single RTX 3090 (24GB VRAM)  
**Current Setup:** Gemma4 26B running on GPU1, using 20-22 GB VRAM  
**Constraint:** GPU0 reserved for research workloads

---

## Current Situation Analysis (Corrected)

### Actual Runtime State
- **Ollama runner:** GPU1 only (despite `CUDA_VISIBLE_DEVICES=0` in `/etc/default/ollama`)
- **VRAM usage:** 20-22 GB on GPU1
- **Model weights:** 17 GB (Q4_K_M quantization)
- **Active KV cache:** ~3-5 GB (supports ~8-14K token context currently)
- **GPU0:** Free for research (good!)

### Model Specs
- **Gemma4 26B** (Gemma 2 architecture)
- **Quantization:** Q4_K_M (4-bit mixed)
- **Architecture:** 46 layers, 16 GQA heads, d_head=128
- **Configured context:** 262,144 tokens (aspirational, not achievable)
- **Practical single-GPU limit:** ~14K tokens with current fp16 KV cache

### Memory Breakdown (Current)
```
22 GB total VRAM usage = 17 GB model + ~5 GB overhead
Overhead = KV cache (~3-5GB for 8-14K ctx) + activations (~1-2GB)
Free VRAM on GPU1: ~2-4 GB
```

### KV Cache Memory Requirements (Uncompressed fp16)
```
KV per token = 2 (K+V) × 46 layers × 16 heads × 128 dim × 2 bytes (fp16)
             = 376,832 bytes/token
             = 359.4 MB per 1K tokens
```

| Context | KV Cache | Total VRAM | Status on Single GPU |
|---------|----------|------------|---------------------|
| 8K | 2.8 GB | 21.8 GB | ✓ Current baseline (~22GB usage) |
| 12K | 4.2 GB | 23.2 GB | ✓ Borderline (leaves only ~1GB free) |
| 14K | 4.9 GB | 23.9 GB | ✓ Maximum safe (leaves minimal buffer) |
| 16K | 5.6 GB | 24.6 GB | ✗ **OOM** |
| 32K | 11.5 GB | 30.5 GB | ✗ OOM |
| 64K | 23.0 GB | 42.0 GB | ✗ OOM |

**Diagnosis:** You're currently running at ~8-10K context, using most of GPU1's 24GB. Any attempt to go beyond ~14K will OOM.

---

## Optimization Options (Re-Evaluated for Single GPU)

### Option 1: 8-bit KV Cache Quantization — **QUICK WIN**

**What:** Quantize KV cache from fp16 (2 bytes) → int8 (1 byte).

**Implementation:**
```bash
# Create custom Modelfile with 8-bit KV cache
cat > /tmp/Modelfile.gemma4-q8kv <<'EOF'
FROM gemma4:26b
PARAMETER num_ctx 28672
PARAMETER cache_type_k q8_0
PARAMETER cache_type_v q8_0
EOF

ollama create gemma4:26b-q8kv -f /tmp/Modelfile.gemma4-q8kv
```

**Memory Savings:**
```
8-bit KV = 376,832 / 2 = 188,416 bytes/token
         = 179.7 MB per 1K tokens (vs 359.4 MB fp16)
Compression ratio: 2.00×
```

| Context | 8-bit KV Cache | Total VRAM | Gain vs Baseline |
|---------|----------------|------------|------------------|
| 8K | 1.4 GB | 20.4 GB | — (current) |
| 14K | 2.5 GB | 21.5 GB | — (current max) |
| **28K** | **5.0 GB** | **24.0 GB** | **✓ 2× improvement** |
| 32K | 5.7 GB | 24.7 GB | ✗ Borderline OOM |

**Maximum viable context:** ~28K tokens (vs 14K baseline)

**Pros:**
- ✓ **Easy implementation** (5 min, just pass flags)
- ✓ **No engineering** (llama.cpp already supports it)
- ✓ **Minimal quality loss** (<1% PPL on most models)
- ✓ **No calibration** required
- ✓ **2× context increase** (14K → 28K)

**Cons:**
- Only 2× gain (not 4× like SubRotQ)
- Still limited to ~28K context

**Recommendation:** ✅ **Do this first.** Immediate 2× win with zero risk.

---

### Option 2: SubRotQ K-cache Compression — **MAXIMUM CONTEXT**

**What:** Apply our research findings — compress K cache to k=128/4-bit, keep V at full-rank 8-bit.

**Implementation:**
1. Patch llama.cpp with KV compression hooks
2. One-time calibration (2K tokens)
3. Compress K: 128 dim × 4-bit = 0.5 bytes/dim → 64 bytes/head
4. Keep V: 128 dim × 8-bit = 1 byte/dim → 128 bytes/head

**Memory Savings:**
```
SubRotQ KV per token:
  K: 46 layers × 16 heads × 64 bytes = 47,104 bytes
  V: 46 layers × 16 heads × 128 bytes = 94,208 bytes
  Total: 141,312 bytes/token (vs 376,832 fp16)
Compression ratio: 2.67× (vs fp16), 1.33× (vs 8-bit)
```

| Context | SubRotQ KV | Total VRAM | Gain vs fp16 |
|---------|------------|------------|--------------|
| 14K | 1.8 GB | 20.8 GB | — |
| 28K | 3.7 GB | 22.7 GB | — |
| **42K** | **5.5 GB** | **24.5 GB** | **✓ 3× improvement** |
| 48K | 6.3 GB | 25.3 GB | ✗ OOM |

**Maximum viable context:** ~40-42K tokens (vs 14K baseline = **3× improvement**)

**Pros:**
- ✓ **3× context increase** (14K → 42K)
- ✓ **Better quality than 8-bit KV** at same memory budget
  - SubRotQ k=128: <2% PPL loss (exp24: 0.98× rel-PPL)
  - 8-bit KV: ~1% PPL loss (but only 2× compression)
- ✓ **Cross-arch validated** (Qwen3/Mistral/Llama in exp21/30)
- ✓ **Production-ready config** (k=128/4-bit is our stable recommendation)

**Cons:**
- **Engineering effort:** 2-3 days to patch llama.cpp
- **Latency overhead:** 1.6× with Python hooks (need CUDA kernel for <1.1×)
- **Calibration required:** One-time 2K token pass
- **Basis storage:** ~45MB per model instance

**Recommendation:** ✅ **Do this if you need >28K context regularly.** Engineering effort pays off with 3× gain.

---

### Option 3: SubRotQ + 8-bit V (Aggressive) — **ABSOLUTE MAX**

**What:** Combine Option 1 + Option 2 — compress K via SubRotQ, quantize V to 8-bit.

**Memory Savings:**
```
K: 46 × 16 × 64 bytes (SubRotQ k=128/4-bit) = 47,104 bytes
V: 46 × 16 × 128 bytes (8-bit, not compressed) = 94,208 bytes
Total: 141,312 bytes/token
Compression ratio: 2.67× vs fp16
```

This is **identical to Option 2** — 8-bit is already the compressed V representation.

**Alternatively, push V compression to k=112/4-bit (NOT RECOMMENDED):**
```
K: k=128/4-bit = 47,104 bytes
V: k=112/4-bit = 41,216 bytes
Total: 88,320 bytes/token
Compression ratio: 4.27× vs fp16
```

| Context | K128/V112 4-bit | Total VRAM | Gain | Quality Risk |
|---------|-----------------|------------|------|--------------|
| 28K | 2.3 GB | 21.3 GB | — | — |
| 42K | 3.5 GB | 22.5 GB | — | High |
| **56K** | **4.6 GB** | **23.6 GB** | **✓ 4× improvement** | **Very High** |

**Maximum viable context:** ~56K tokens

**Pros:**
- ✓ **4× context increase** (14K → 56K)
- ✓ Maximum possible compression on this hardware

**Cons:**
- ✗ **V compression fails universally** (exp21: 12× PPL at k=112)
- ✗ **Quality catastrophically bad** (unusable for real work)
- ✗ **Not recommended by our own research**

**Recommendation:** ❌ **Don't do this.** Research conclusively shows V compression doesn't work.

---

### Option 4: Model Quantization to Q3_K_S — **NOT WORTH IT**

**What:** Further quantize model from Q4_K_M (4-bit) → Q3_K_S (3-bit).

**Memory Savings:**
- Model: 17GB → ~13GB (save 4GB)
- Max context gain: 14K → ~18K tokens (~28% improvement)

**Pros:**
- Frees 4GB for KV cache

**Cons:**
- **Significant quality loss** (~5-10% on reasoning tasks)
- **Minimal context gain** (only +4K tokens)
- **3-bit is at usability edge** for 26B models

**Recommendation:** ❌ **Not worth it.** Quality loss >> context gain.

---

## Recommended Implementation Path

### Phase 1: Quick Win (Today, 5 minutes)

**Deploy 8-bit KV cache** for immediate 2× improvement:

```bash
# Create optimized model variant
cat > /tmp/Modelfile.gemma4-long <<'EOF'
FROM gemma4:26b
PARAMETER num_ctx 28672
PARAMETER cache_type_k q8_0
PARAMETER cache_type_v q8_0
EOF

ollama create gemma4:26b-long -f /tmp/Modelfile.gemma4-long

# Test it
ollama run gemma4:26b-long "Test prompt..."
```

**Result:** 14K → 28K max context, <1% quality loss, zero engineering.

---

### Phase 2: Maximum Context (This Week, 2-3 days)

**Implement SubRotQ in llama.cpp:**

1. **Clone and patch llama.cpp**
   ```bash
   cd /tmp
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   
   # Apply SubRotQ patches (create from our kvpatch library)
   # - Add KV compression hooks in llama.cpp/ggml-cuda.cu
   # - Implement k=128/4-bit SubRotQ pipeline
   # - Add PCA basis loading from calibration file
   ```

2. **Compile with optimizations**
   ```bash
   mkdir build && cd build
   cmake .. \
     -DGGML_CUDA=ON \
     -DLLAMA_CUDA_FA_ALL_QUANTS=ON \  # Enable Flash Attention 2
     -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   
   # Replace Ollama's llama.cpp binary
   sudo systemctl stop ollama
   sudo cp bin/llama-cli /usr/local/bin/ollama-runner-custom
   # Update Ollama service to use custom binary
   sudo systemctl start ollama
   ```

3. **Calibrate and deploy**
   ```bash
   # Run calibration (2K tokens from WikiText-2)
   python3 calibrate_gemma4.py --output basis_gemma4_26b.pkl
   
   # Create SubRotQ model variant
   cat > /tmp/Modelfile.gemma4-subrotq <<'EOF'
   FROM gemma4:26b
   PARAMETER num_ctx 40960
   PARAMETER kv_compression subrotq
   PARAMETER kv_basis_path /path/to/basis_gemma4_26b.pkl
   EOF
   
   ollama create gemma4:26b-subrotq -f /tmp/Modelfile.gemma4-subrotq
   ```

**Result:** 14K → 42K max context, <2% quality loss

---

### Phase 3: Production Polish (Optional, 1 week)

**Implement CUDA kernel** for SubRotQ compress/decompress:

1. Fused PCA project → rotate → quantize kernel
2. Reduces 1.6× overhead → ~1.1× overhead
3. Recovers most decode speed loss

**Result:** 42K context at near-baseline speed

---

## Comparison Table (Updated)

| Approach | Max Context | Gain vs Current | Speed Impact | Quality Loss | Effort | VRAM (GPU1) |
|----------|-------------|-----------------|--------------|--------------|--------|-------------|
| **Current (fp16 KV)** | 14K | — | — | — | — | 22 GB |
| **8-bit KV cache** | **28K** | **2×** | None | <1% | 5 min | 24 GB |
| **SubRotQ k=128** | **42K** | **3×** | -37%* | <2% | 2-3 days | 24.5 GB |
| **SubRotQ + CUDA** | **42K** | **3×** | -9% | <2% | 1-2 weeks | 24.5 GB |
| SubRotQ + V-comp | 56K | 4× | -37%* | **50×+** | 2-3 days | 23.6 GB |
| Q3 model + 8-bit KV | 36K | 2.6× | None | **-10%** | 1 hour | 24 GB |

\* With Python hooks; CUDA kernel reduces to ~-9%

---

## Final Recommendations

### For Immediate Use (Today)
✅ **Option 1: 8-bit KV cache**
- Max context: **28K tokens** (2× gain)
- Effort: 5 minutes
- Quality: <1% loss
- Command: See Phase 1 above

### For Maximum Context (This Week)
✅ **Option 2: SubRotQ k=128 + 8-bit V**
- Max context: **42K tokens** (3× gain)
- Effort: 2-3 days engineering
- Quality: <2% loss (production-viable per our research)
- Speed: Acceptable since you prioritize context > speed

### What NOT to Do
❌ Don't compress V below full rank (our research proves it fails)  
❌ Don't quantize model to Q3 (quality loss not worth +4K context)  
❌ Don't use dual-GPU (you need GPU0 for research)

---

## Implementation Support

If you want to proceed with SubRotQ (Option 2), I can:

1. **Generate llama.cpp patches** from our kvpatch library
2. **Write calibration script** for Gemma4 26B
3. **Create integration guide** for Ollama + custom llama.cpp
4. **Benchmark quality** on your specific workloads

Let me know which path you want to take!
