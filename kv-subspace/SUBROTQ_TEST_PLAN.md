# SubRotQ Testing & Calibration Plan

## Current Status
- ✅ llama.cpp with SubRotQ implemented (identity basis)
- ✅ CLI flags working (`--subrotq`, `--subrotq-rank`, `--subrotq-bits`)
- ✅ CUDA kernels active
- ❌ No real PCA basis yet (using identity U=I, mean=0, scale=1)

## Problem: Calibration Chicken-and-Egg

**Goal:** Get SubRotQ working on Gemma4 26B (single RTX 3090, 24GB)

**Challenge:** 
- Gemma4 26B GGUF is already quantized (Q4_K_M, 17GB)
- llama.cpp doesn't expose KV cache for extraction
- Unquantized Gemma4 (~52GB) won't fit on 24GB GPU for PyTorch calibration
- Can't calibrate without running the model

## Solution: Two-Phase Approach

### Phase A: Validate SubRotQ Pipeline (Now)
**Use TinyLlama-1.1B to verify the implementation works end-to-end**

1. Download TinyLlama-1.1B GGUF (~600MB, fits easily)
2. Run llama-cli with `--subrotq` (identity basis)
3. Verify:
   - SubRotQ init logs appear
   - Compression/decompression hooks fire
   - Output is coherent (identity basis is lossless)
   - No crashes or CUDA errors
4. **This proves the pipeline works**

### Phase B: Real Calibration for Gemma4 (After validation)
**Two options:**

#### Option B1: Calibrate on Smaller Similar Model ✅ RECOMMENDED
- Use Qwen3-14B (already have it, works with our code)
- Collect K vectors using existing `collect_kvs_for_basis()` from exp24
- Compute PCA basis (128-rank per layer/head)
- **Assumption:** Gemma4 and Qwen3 have similar K-cache structure
  - Both use GQA (8 KV heads)
  - Both have d_head=128
  - PCA basis should transfer reasonably well
- Save basis to `.bin` file
- Modify llama.cpp to load basis file instead of identity init
- Test on Gemma4 26B

**Pros:** 
- Uses proven calibration code
- Fast (we already have Qwen3-14B loaded)
- Can iterate quickly

**Cons:**
- Cross-model transfer may reduce quality slightly
- Not perfect Gemma4-specific basis

#### Option B2: Calibrate Directly on Gemma4 GGUF
- Modify llama.cpp to add a `--calibrate-subrotq` mode
- Hook into KV cache write path to dump K vectors to file
- Run on WikiText-2 first 2K tokens
- Compute PCA offline from dumped vectors
- Save basis and reload

**Pros:**
- Gemma4-specific basis (optimal quality)

**Cons:**
- Requires modifying llama.cpp more extensively
- Slower development cycle

## Recommendation: A → B1

1. **Today:** Validate with TinyLlama (30 min)
2. **Next:** Calibrate using Qwen3-14B (2 hours)
3. **Deploy:** Test on Gemma4 26B with transferred basis
4. **If needed:** Refine with option B2 for Gemma4-specific basis

## Phase A Implementation (Next Steps)

```bash
# 1. Download TinyLlama GGUF
cd /tmp
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# 2. Test SubRotQ with identity basis
/tmp/llama.cpp/build/bin/llama-cli \
  -m /tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --subrotq \
  --subrotq-rank 128 \
  --subrotq-bits 4 \
  -p "Once upon a time" \
  -n 50 \
  --verbose

# 3. Check logs for:
# - "Initializing SubRotQ with k=128, n_bits=4"
# - "SubRotQ: decompressing layer X head Y"
# - "SubRotQ: compressing X vectors"

# 4. Verify output is coherent
```

## Phase B1 Implementation (After validation)

```bash
# 1. Use existing Qwen3-14B calibration
cd /home/petty/pruning-research/kv-subspace
python3 calibrate_qwen3_for_gemma4.py  # New script, adapts exp24

# 2. Saves: qwen3_subrotq_k128.bin

# 3. Modify llama.cpp init_subrotq() to load from file instead of identity

# 4. Rebuild llama.cpp

# 5. Test on Gemma4:
/tmp/llama.cpp/build/bin/llama-cli \
  -m /usr/share/ollama/.ollama/models/blobs/sha256-7121... \
  --subrotq \
  --subrotq-basis qwen3_subrotq_k128.bin \
  --ctx-size 65536 \
  -p "Explain quantum computing" \
  -n 100

# 6. Monitor GPU memory - should see context scale beyond 32K
```

## Success Criteria

**Phase A (Validation):**
- ✅ TinyLlama runs without crashes
- ✅ Debug logs show SubRotQ init/compress/decompress
- ✅ Output is coherent
- ✅ Identity basis confirmed lossless

**Phase B1 (Calibration):**
- ✅ Qwen3 basis file generated (~50MB for 40 layers × 8 heads)
- ✅ llama.cpp loads basis file successfully
- ✅ Gemma4 26B fits in 24GB with 60K+ context
- ✅ Quality acceptable (subjective prompt test)
- ✅ Memory savings measurable (VRAM drops or context increases)

## Timeline Estimate

- **Phase A:** 30-60 minutes (download + test)
- **Phase B1:** 2-3 hours (calibration script + llama.cpp mods + testing)
- **Total:** ~4 hours to working Gemma4 SubRotQ

Ready to proceed with Phase A (TinyLlama validation)?
