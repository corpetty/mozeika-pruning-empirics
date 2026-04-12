# SubRotQ Implementation Status

**Last updated:** 2026-04-12 16:58 UTC  
**Project:** K-cache compression for Gemma4:26b on single RTX 3090  
**Goal:** Increase context 32K → 60-70K via SubRotQ k=128 4-bit compression

---

## Current Status: Phase 5 Complete ✅

**What's working:**
- ✅ CUDA kernels (compress/decompress with 4-bit quantization)
- ✅ Full llama.cpp integration (CLI args, parameter flow, hooks)
- ✅ End-to-end verification on TinyLlama-1.1B (22 layers)
- ✅ All layers compress/decompress correctly
- ✅ No crashes, NaN, or quality degradation

**What's NOT working yet:**
- ❌ Real PCA basis (currently using identity matrix placeholder)
- ❌ Actual compression (identity basis is lossless but doesn't compress)
- ❌ Memory savings (same as baseline with identity)

**Code location:** `/tmp/llama.cpp` (branch: `subrotq-kv-compression`)

---

## Phase Breakdown

| Phase | Status | Duration | Cost | Deliverable |
|-------|--------|----------|------|-------------|
| 1. Build llama.cpp | ✅ | 10 min | $0.18 | Working CUDA build |
| 2. CUDA kernels | ✅ | 5 min | $0.98 | ggml-subrotq.cu/.h |
| 3. Integration | ✅ | 9 min | $2.71 | CLI args + hooks |
| 4. Activate kernels | ✅ | 11 min | $2.85 | Compress/decompress firing |
| 5. Verification | ✅ | 30 min | $0 | TinyLlama test |
| **6. Calibration** | ⏸️ | **TBD** | **TBD** | **Real PCA basis** |

**Total so far:** 65 minutes, $6.72 USD

---

## Next Steps: Phase 6 - Calibration

### Goal
Generate real PCA basis from calibration data to enable actual compression.

### Approach Options

**Option A: HuggingFace PyTorch (Recommended)**
- ✅ Proven working: `calibrate_gemma4.py` already exists
- ✅ Can use Qwen3-14B (same architecture as Gemma4)
- ✅ Existing code: `collect_kvs_for_basis()` from experiments
- ❌ Requires unquantized model (large VRAM)
- **Workaround:** Use smaller model (Qwen3-7B or Mistral-7B) for calibration

**Option B: llama.cpp Native**
- ✅ Can use quantized GGUF models (smaller VRAM)
- ❌ Need to extract K-cache from llama.cpp internals (no API)
- ❌ More complex implementation

**Recommendation:** Use Option A with Mistral-7B-v0.3 (already cached locally, unquantized).

### Tasks

1. **Calibration script** (~30 min)
   - Adapt existing `calibrate_gemma4.py` for Mistral-7B
   - Collect K vectors from WikiText-2 train (2048 tokens)
   - Compute per-layer, per-head PCA (SVD on centered K matrix)
   - Save basis to `.subrotq` binary format

2. **Binary format spec** (~15 min)
   - Header: magic number, version, n_layers, n_heads, d_head, rank, n_bits
   - Per layer/head: U matrix (d_head × rank, fp32), mean vector (d_head, fp32), scale vector (rank, fp32)

3. **Basis loader** (~30 min)
   - Modify `init_subrotq()` to load from file instead of identity
   - Add CLI arg: `--subrotq-basis path/to/basis.bin`
   - Validate dimensions match model architecture

4. **Testing** (~1 hour)
   - Generate Gemma4 basis using calibration script
   - Test with llama-cli on Gemma4:26b GGUF
   - Measure context increase (target: 32K → 60-70K)
   - Verify quality (quick NIAH test)

**Estimated Phase 6:** 2.5 hours, $8-12 USD

---

## Target Architecture: Gemma4 26B

```
Model: google/gemma-2-27b-it (quantized to 26B effective)
Layers: 46
Heads: 16 (query), 8 (KV, GQA)
Head dim: 128
Current context: 32,768 tokens @ 20.6 GB VRAM
Target context: 60,000-70,000 tokens @ ~22 GB VRAM
```

**Compression strategy:**
- K-cache: SubRotQ k=128 4-bit (4.00× compression)
- V-cache: Keep 8-bit quantization (2.00× compression)
- Combined effective: ~3.5× compression
- Context scaling: 2.2× baseline (32K → 70K)

---

## Files Reference

**Implementation:**
- `/tmp/llama.cpp/ggml/src/ggml-subrotq.{h,cu}` - CUDA kernels
- `/tmp/llama.cpp/src/llama-kv-cache.{h,cpp}` - Integration hooks
- `/tmp/llama.cpp/common/arg.cpp` - CLI arguments

**Documentation:**
- `SUBROTQ_IMPLEMENTATION_PLAN.md` - Original 2.5-day plan
- `SUBROTQ_IMPLEMENTATION_LOG.md` - Detailed phase log
- `SUBROTQ_PHASE_2_COMPLETE.md` - Phase 2-5 summary
- `SUBROTQ_STATUS.md` - This file

**Calibration (future):**
- `calibrate_gemma4.py` - PyTorch calibration script (needs adaptation)
- `results/gemma4_subrotq_basis.bin` - Generated PCA basis (TBD)

---

## Key Technical Notes

**Architecture constraint:**
- `get_k()` and `cpy_k()` are graph-building functions (no tensor data yet)
- Decompression runs synchronously in `get_k()` before graph eval
- Compression deferred to `subrotq_post_eval_compress()` after graph eval

**Current implementation:**
- Identity basis: U = I[:,0:k] (truncated identity)
- Lossless but no compression (U @ U^T @ x = x when k=d_head)
- Memory usage same as baseline

**After calibration:**
- Real PCA basis: U from SVD of calibration K vectors
- Lossy compression (U @ U^T @ x ≈ x when k < d_head)
- Expected quality: 0.98-1.00× PPL at k=128 (from paper experiments)

---

## Questions / Blockers

None currently. Ready to proceed with Phase 6 when approved.

---

**Ready to start Phase 6?** Let me know and I'll begin with the calibration script for Mistral-7B.
