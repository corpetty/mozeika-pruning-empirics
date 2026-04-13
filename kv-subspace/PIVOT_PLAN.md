# SubRotQ Implementation — Pivot Plan

## Current Situation
- ✅ SubRotQ fully implemented in llama.cpp (CUDA kernels, basis loader, CLI args)
- ✅ TinyLlama validation successful (end-to-end compression working)
- ✅ Calibration pipeline working (Mistral k=128 basis generated)
- ❌ Ollama Gemma4 GGUF incompatible (tensor count mismatch)
- ❌ Ollama binary statically compiled (can't inject shared library)

## The Problem
**Goal:** 3× context scaling on Gemma4-26B via SubRotQ k=128/4-bit
**Blocker:** Can't get Gemma4 into llama.cpp in a usable format

## Options

### Option A: Fix Gemma4 GGUF (High Complexity)
1. Update llama.cpp to latest (may have Gemma4 fixes)
2. Or: Convert Ollama's Gemma4 weights to standard GGUF manually
3. Or: Find a pre-quantized Gemma4 GGUF that llama.cpp accepts

**Risk:** llama.cpp may not support this Gemma4 variant at all
**Time:** Unknown (could be days of debugging)

### Option B: Recompile Ollama with SubRotQ (Massive Effort)
1. Clone Ollama source
2. Replace bundled llama.cpp with our SubRotQ fork
3. Rebuild entire Ollama binary
4. Test with Gemma4

**Risk:** Ollama build complexity, ongoing maintenance burden
**Time:** 1-2 days minimum

### Option C: Use Gemma-2-27B GGUF Instead (Proven Path) ⭐
1. Download google/gemma-2-27b Q4_K_M GGUF from HuggingFace
2. Generate Gemma-2 specific PCA basis (or reuse Mistral cross-arch)
3. Test SubRotQ end-to-end with llama-cli
4. Measure context scaling and VRAM savings
5. **Prove SubRotQ works on a production-scale model**

**Advantages:**
- Gemma-2-27B is same family as Gemma4
- HuggingFace GGUFs are llama.cpp-compatible
- We can demonstrate the full value proposition
- Clean, reproducible results for the paper

**Disadvantages:**
- Not the exact model you're using in Ollama (Gemma4 vs Gemma-2)
- Need to download 17GB GGUF

## Recommendation: **Option C**

**Why:** 
1. We've already invested $10+ in SubRotQ implementation
2. The core research question is "does SubRotQ enable 3× context scaling?"
3. Gemma-2-27B is close enough to answer that question
4. We can always revisit Ollama integration later if results are good

**What Success Looks Like:**
- Gemma-2-27B running at 60-70K context on single RTX 3090
- SubRotQ k=128/4-bit compression working end-to-end
- Coherent text generation at long context
- Measurable VRAM savings (~4× KV cache reduction)
- **Publishable results for the paper**

## Next Steps (Option C)

1. **Download Gemma-2-27B GGUF** (~17GB, 30 min)
   ```bash
   cd /tmp
   wget https://huggingface.co/google/gemma-2-27b-it-GGUF/resolve/main/gemma-2-27b-it.Q4_K_M.gguf
   ```

2. **Generate Gemma-2 PCA Basis** (or reuse Mistral k=128)
   - Try Mistral basis first (already validated cross-arch in exp30)
   - If quality degrades, generate Gemma-2 specific basis

3. **Test SubRotQ**
   ```bash
   CUDA_VISIBLE_DEVICES=0 /tmp/llama.cpp/build/bin/llama-cli \
     -m /tmp/gemma-2-27b-it.Q4_K_M.gguf \
     -p "Explain quantum entanglement in detail (at least 500 words):" \
     -n 500 \
     -c 65536 \
     --subrotq \
     --subrotq-rank 128 \
     --subrotq-bits 4 \
     --subrotq-basis results/subrotq_basis_mistral7b_k128.bin
   ```

4. **Measure Context Scaling**
   - Test at 16K, 32K, 48K, 64K, 80K tokens
   - Monitor VRAM usage with nvidia-smi
   - Record when OOM occurs
   - Compare vs baseline (no SubRotQ)

5. **Document Results**
   - Context scaling achieved (e.g., 32K → 70K = 2.2×)
   - VRAM savings (e.g., 22GB vs 24GB native)
   - Quality assessment (sample outputs)
   - Add to paper as real-world validation

## Fallback to Ollama Later

If Gemma-2 results are compelling, we can:
1. Contact Ollama maintainers about SubRotQ integration
2. Or: Use Gemma-2 in production via llama.cpp instead of Ollama
3. Or: Wait for llama.cpp to support Gemma4 architecture natively

---

**Decision Point:** Proceed with Option C (Gemma-2-27B)?
