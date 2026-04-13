# SubRotQ Phase 8: Gemma4-26B GGUF Extraction and Basis Calibration

**Context:**
- We've built SubRotQ (Subspace Rotation Quantization) KV-cache compression into llama.cpp
- Successfully validated end-to-end on TinyLlama (d_head=64, k=64/4-bit)
- Target: Gemma4-26B Q4_K_M currently running in Ollama on GPU1
- Gemma4 architecture: d_head=176, 30 layers, 16 heads — perfect for k=128 compression

**Goal:** 3× context scaling (32K → ~70K tokens) on single RTX 3090 via SubRotQ k=128/4-bit compression

**Your tasks:**

1. **Extract Gemma4 GGUF from Ollama**
   - Ollama model: `gemma4:26b` (hash: 5571076f3d70)
   - Blob path from modelfile: `/usr/share/ollama/.ollama/models/blobs/sha256-7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df`
   - Copy to: `/tmp/gemma4-26b.gguf`
   - Verify it's valid GGUF (test load with llama-cli)

2. **Generate Gemma4 PCA basis via calibration**
   - Use existing script: `calibrate_subrotq_basis.py`
   - Challenge: Gemma models need HuggingFace auth (gated)
   - **Alternative approach:** Use Mistral-7B-v0.3 basis as cross-architecture proxy
     - Already validated in exp30: k=112/4-bit works cross-arch
     - Mistral cached at ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3
     - Basis already exists: `results/subrotq_basis_mistral7b_k128.bin`
   - If you want Gemma-specific: try google/gemma-2-9b or google/gemma-2-27b (needs HF_TOKEN)
   - Rank: 128, Bits: 4, Calibration tokens: 2048
   - Output: `results/subrotq_basis_gemma4_k128.bin` (or reuse Mistral if cross-arch)

3. **Test SubRotQ on Gemma4 GGUF**
   - llama.cpp build: `/tmp/llama.cpp/build/bin/llama-cli`
   - Run with SubRotQ:
     ```bash
     CUDA_VISIBLE_DEVICES=0 /tmp/llama.cpp/build/bin/llama-cli \
       -m /tmp/gemma4-26b.gguf \
       -p "Write a detailed explanation of quantum entanglement" \
       -n 100 \
       --subrotq \
       --subrotq-rank 128 \
       --subrotq-bits 4 \
       --subrotq-basis results/subrotq_basis_mistral7b_k128.bin
     ```
   - Verify: compression logs, coherent output, no crashes

**Constraints:**
- GPU0 ONLY (export CUDA_VISIBLE_DEVICES=0 before all GPU ops)
- venv: `/home/petty/pruning-research/venv` (source before Python)
- Never delete files
- If stuck on Gemma access, use Mistral basis (already proven cross-arch compatible)

**Deliverable:**
Brief report with:
- GGUF extraction status + size
- Basis choice (Mistral reuse or fresh Gemma)
- Test output snippet (20-30 tokens)
- Issues + workarounds
