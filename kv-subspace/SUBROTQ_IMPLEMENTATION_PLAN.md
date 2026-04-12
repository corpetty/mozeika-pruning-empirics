# SubRotQ Implementation Plan for Ollama/llama.cpp

**Goal:** 42K max context on single GPU (3× baseline) with <2% quality loss  
**Approach:** Patch llama.cpp with SubRotQ K-cache compression  
**Effort Estimate:** 2-3 days  
**Status:** Planning → Implementation → Testing → Deployment

---

## Architecture Overview

### Current State (8-bit KV cache)
```
Model weights (Q4_K_M):     17 GB
KV cache (8-bit, 28K ctx):  ~5 GB (188,416 bytes/token)
Activations + overhead:     ~2 GB
─────────────────────────────────
Total:                      ~24 GB (GPU1 only)
```

### Target State (SubRotQ K + 8-bit V)
```
Model weights (Q4_K_M):     17 GB
K cache (SubRotQ k=128/4-bit): ~2.5 GB (94,208 bytes/token at 42K ctx)
V cache (8-bit, 42K ctx):      ~3.8 GB (94,208 bytes/token at 42K ctx)
Activations + overhead:         ~2 GB
─────────────────────────────────
Total:                         ~25.3 GB → fits on 24GB GPU with slight overflow*
```

\* May need to reduce to 40K context for safe margin, or use activation checkpointing

### Memory Calculation
```python
# Gemma4 26B: 46 layers, 16 KV heads, d_head=128

# Current 8-bit KV (both K and V at 8-bit):
kv_per_token = 2 (K+V) × 46 layers × 16 heads × 128 dim × 1 byte
             = 188,416 bytes/token
             = 179.7 MB per 1K tokens
Max at 24GB: ~28K tokens

# SubRotQ K=128/4-bit + V=8-bit:
k_per_token = 46 layers × 16 heads × (128 dim × 0.5 bytes)  # 4-bit = 0.5 byte/dim
            = 47,104 bytes/token
v_per_token = 46 layers × 16 heads × (128 dim × 1 byte)
            = 94,208 bytes/token
total_per_token = 141,312 bytes/token
                = 134.8 MB per 1K tokens
Max at 24GB: ~44K tokens (theoretical)
Safe target: ~40-42K tokens
```

**Compression ratio:** 188,416 / 141,312 = **1.33× additional gain** over 8-bit alone

---

## Implementation Phases

### Phase 1: Setup & Environment (Day 1 Morning)

**1.1 Clone and Build llama.cpp**
```bash
cd /tmp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git checkout master  # or specific stable tag

mkdir build && cd build
cmake .. \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DLLAMA_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
```

**1.2 Verify baseline build**
```bash
# Test with a small model first
./bin/llama-cli --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-14B-AWQ/snapshots/.../model.gguf \
  --prompt "Test" -n 10

# Check that CUDA backend works
./bin/llama-cli --help | grep -i cuda
```

**1.3 Create development branch**
```bash
git checkout -b subrotq-kv-compression
```

---

### Phase 2: Core SubRotQ Implementation (Day 1 Afternoon → Day 2)

**2.1 Add SubRotQ compression primitives**

Create `ggml-subrotq.h` and `ggml-subrotq.cu`:

```c
// ggml-subrotq.h
#pragma once

#include "ggml.h"
#include <stdint.h>

// SubRotQ compression parameters
struct subrotq_params {
    int32_t k;              // Subspace rank (e.g., 128)
    int32_t n_bits;         // Quantization bits (4 or 8)
    int32_t d_head;         // Head dimension (128 for Gemma4)
    float * U;              // PCA basis (d_head × k), column-major
    float * mean;           // Mean vector (d_head)
    int32_t * codebook;     // Uniform quantization codebook (unused for now)
};

// Compress K vector from fp16 → SubRotQ
void subrotq_compress_k(
    const ggml_fp16_t * k_fp16,  // Input: [d_head]
    uint8_t * k_compressed,      // Output: [k * n_bits / 8] packed
    const struct subrotq_params * params
);

// Decompress K vector SubRotQ → fp16
void subrotq_decompress_k(
    const uint8_t * k_compressed,
    ggml_fp16_t * k_fp16,
    const struct subrotq_params * params
);

// Fit PCA basis from calibration K vectors
void subrotq_fit_basis(
    const ggml_fp16_t * k_vectors,  // [n_samples × d_head]
    int32_t n_samples,
    struct subrotq_params * params  // Fill in U and mean
);
```

**2.2 Implement CUDA kernels**

`ggml-subrotq.cu`:
```cuda
#include "ggml-subrotq.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel: Project K to subspace, quantize to 4-bit
__global__ void subrotq_compress_kernel(
    const half * k_fp16,        // [d_head]
    uint8_t * k_compressed,     // [k * 4 / 8] = [k/2] bytes for 4-bit
    const float * U,            // [d_head × k]
    const float * mean,         // [d_head]
    int32_t d_head,
    int32_t k
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= k) return;
    
    // 1. Center: x_centered = k - mean
    // 2. Project: z[tid] = dot(x_centered, U[:, tid])
    float z = 0.0f;
    for (int i = 0; i < d_head; i++) {
        float x_i = __half2float(k_fp16[i]) - mean[i];
        z += x_i * U[i * k + tid];  // Column-major U
    }
    
    // 3. Quantize to 4-bit: map z ∈ [-bound, bound] → [0, 15]
    //    Use per-dimension min/max scaling (stored separately)
    //    For simplicity, use uniform [-3σ, 3σ] clipping
    const float sigma = 1.0f;  // Placeholder, compute from calibration
    const float bound = 3.0f * sigma;
    float z_clipped = fminf(fmaxf(z, -bound), bound);
    int32_t z_quant = (int32_t)roundf((z_clipped + bound) / (2.0f * bound) * 15.0f);
    
    // 4. Pack two 4-bit values per byte
    int byte_idx = tid / 2;
    int bit_offset = (tid % 2) * 4;
    
    if (tid % 2 == 0) {
        // Lower 4 bits
        k_compressed[byte_idx] = (k_compressed[byte_idx] & 0xF0) | (z_quant & 0x0F);
    } else {
        // Upper 4 bits
        k_compressed[byte_idx] = (k_compressed[byte_idx] & 0x0F) | ((z_quant & 0x0F) << 4);
    }
}

// Host wrapper
void subrotq_compress_k(
    const ggml_fp16_t * k_fp16,
    uint8_t * k_compressed,
    const struct subrotq_params * params
) {
    int threads = 256;
    int blocks = (params->k + threads - 1) / threads;
    
    subrotq_compress_kernel<<<blocks, threads>>>(
        (const half *)k_fp16,
        k_compressed,
        params->U,
        params->mean,
        params->d_head,
        params->k
    );
    cudaDeviceSynchronize();
}

// TODO: Implement subrotq_decompress_kernel (reverse process)
// TODO: Implement subrotq_fit_basis (PCA via cuSOLVER or Eigen)
```

**2.3 Integrate into llama.cpp KV cache**

Modify `llama.cpp` (KV cache storage):

```cpp
// In llama_kv_cache struct (llama.cpp)
struct llama_kv_cache {
    // ... existing fields ...
    
    // SubRotQ compression
    bool use_subrotq;
    struct subrotq_params subrotq_k;  // Per-layer, per-head params
    ggml_tensor * k_compressed;       // Compressed K storage
    ggml_tensor * v_quantized;        // 8-bit V storage (existing)
};
```

Modify KV cache write (in `llama_decode_internal`):

```cpp
// When writing K cache (simplified pseudocode)
if (kv_cache.use_subrotq) {
    // Compress K on-the-fly
    ggml_fp16_t * k_fp16 = ggml_get_data_f16(k_tensor);
    uint8_t * k_compressed_dst = ggml_get_data(kv_cache.k_compressed) + offset;
    
    subrotq_compress_k(k_fp16, k_compressed_dst, &kv_cache.subrotq_k);
} else {
    // Standard path (copy fp16 or quantized K)
    // ...
}
```

Modify KV cache read (in attention computation):

```cpp
// When reading K cache for attention
if (kv_cache.use_subrotq) {
    // Decompress K on-the-fly
    uint8_t * k_compressed_src = ggml_get_data(kv_cache.k_compressed) + offset;
    ggml_fp16_t * k_fp16_tmp = allocate_temp_buffer(d_head);
    
    subrotq_decompress_k(k_compressed_src, k_fp16_tmp, &kv_cache.subrotq_k);
    
    // Use k_fp16_tmp in attention (QK^T)
    // ...
} else {
    // Standard path
    // ...
}
```

---

### Phase 3: Calibration & Basis Fitting (Day 2 Afternoon)

**3.1 Calibration Script**

Create `calibrate_subrotq.py`:

```python
#!/usr/bin/env python3
import sys
sys.path.append('/home/petty/pruning-research/kv-subspace')

import torch
import numpy as np
from collect import get_model_and_tokenizer, get_wikitext2_tokens
from compress import fit_pca
import pickle

def calibrate_gemma4_subrotq(k=128, n_bits=4, calib_tokens=2048):
    """
    Fit SubRotQ PCA bases for Gemma4 26B.
    Saves basis to gemma4_subrotq_k{k}_basis.pkl
    """
    print(f"Calibrating SubRotQ: k={k}, n_bits={n_bits}, calib_tokens={calib_tokens}")
    
    # Load model
    model, tokenizer = get_model_and_tokenizer(
        "gemma4:26b",  # Ollama model name → resolve to HF path
        device_map="cuda:1"  # Use GPU1 (GPU0 reserved)
    )
    
    # Get calibration data (WikiText-2 train)
    calib_text = get_wikitext2_tokens(tokenizer, split='train', max_tokens=calib_tokens)
    input_ids = tokenizer(calib_text, return_tensors='pt').input_ids.to('cuda:1')
    
    # Collect K vectors
    print("Collecting K vectors from calibration pass...")
    kv_store = {}
    hooks = []
    
    def hook_k_proj(layer_idx, head_idx):
        def _hook(module, input, output):
            # output shape: [batch, seq_len, n_heads * d_head]
            k_proj = output[0].detach()  # [seq_len, n_heads * d_head]
            k_proj = k_proj.view(-1, 16, 128)  # [seq_len, 16 heads, 128 dim]
            k_head = k_proj[:, head_idx, :].cpu().numpy()  # [seq_len, 128]
            
            key = (layer_idx, head_idx)
            if key not in kv_store:
                kv_store[key] = []
            kv_store[key].append(k_head)
        return _hook
    
    # Register hooks on all layers/heads
    for layer_idx in range(46):  # Gemma4: 46 layers
        attn = model.model.layers[layer_idx].self_attn
        for head_idx in range(16):  # 16 KV heads
            hook = attn.k_proj.register_forward_hook(hook_k_proj(layer_idx, head_idx))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        model(input_ids)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Fit PCA bases
    print("Fitting PCA bases...")
    bases = {}
    for (layer_idx, head_idx), k_list in kv_store.items():
        K_np = np.concatenate(k_list, axis=0)  # [total_tokens, 128]
        
        # Fit PCA
        mean = K_np.mean(axis=0)
        U_k, explained_var = fit_pca(K_np - mean, k)  # U_k: [128, k]
        
        bases[(layer_idx, head_idx)] = {
            'U': U_k.astype(np.float32),
            'mean': mean.astype(np.float32),
            'explained_var': explained_var,
            'k': k,
            'n_bits': n_bits,
            'd_head': 128
        }
        
        print(f"  Layer {layer_idx} Head {head_idx}: "
              f"explained variance = {explained_var.sum():.4f}")
    
    # Save to pickle
    output_path = f'gemma4_subrotq_k{k}_basis.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(bases, f)
    
    print(f"Saved basis to {output_path}")
    print(f"Total file size: {os.path.getsize(output_path) / 1e6:.1f} MB")

if __name__ == '__main__':
    calibrate_gemma4_subrotq(k=128, n_bits=4, calib_tokens=2048)
```

**3.2 Convert Basis to llama.cpp Format**

Create `export_basis_to_gguf.py`:

```python
#!/usr/bin/env python3
import pickle
import struct

def export_subrotq_basis_to_binary(pkl_path, output_path):
    """
    Convert Python pickle basis → binary format for llama.cpp
    
    Format (per layer/head):
        - 4 bytes: layer_idx (int32)
        - 4 bytes: head_idx (int32)
        - 4 bytes: k (int32)
        - 4 bytes: d_head (int32)
        - d_head × 4 bytes: mean (float32 array)
        - d_head × k × 4 bytes: U (float32 column-major)
    """
    with open(pkl_path, 'rb') as f:
        bases = pickle.load(f)
    
    with open(output_path, 'wb') as out:
        # Header: magic number + version
        out.write(b'SUBQ')  # Magic
        out.write(struct.pack('I', 1))  # Version
        out.write(struct.pack('I', len(bases)))  # Number of entries
        
        for (layer_idx, head_idx), params in bases.items():
            # Write layer/head metadata
            out.write(struct.pack('iiii', layer_idx, head_idx, params['k'], params['d_head']))
            
            # Write mean vector
            mean = params['mean']  # [d_head]
            out.write(mean.tobytes())
            
            # Write U matrix (column-major)
            U = params['U']  # [d_head, k]
            out.write(U.tobytes())
    
    print(f"Exported basis to {output_path}")

if __name__ == '__main__':
    export_subrotq_basis_to_binary(
        'gemma4_subrotq_k128_basis.pkl',
        'gemma4_subrotq_k128_basis.bin'
    )
```

---

### Phase 4: llama.cpp Integration (Day 2 Evening → Day 3)

**4.1 Load Basis in llama.cpp**

Add basis loading to `llama.cpp`:

```cpp
// llama.cpp: Load SubRotQ basis from binary file
bool llama_kv_cache_load_subrotq_basis(
    struct llama_kv_cache * cache,
    const char * basis_path
) {
    FILE * f = fopen(basis_path, "rb");
    if (!f) return false;
    
    // Read header
    char magic[4];
    uint32_t version, n_entries;
    fread(magic, 1, 4, f);
    fread(&version, 4, 1, f);
    fread(&n_entries, 4, 1, f);
    
    if (memcmp(magic, "SUBQ", 4) != 0 || version != 1) {
        fclose(f);
        return false;
    }
    
    // Allocate GPU memory for bases
    for (uint32_t i = 0; i < n_entries; i++) {
        int32_t layer_idx, head_idx, k, d_head;
        fread(&layer_idx, 4, 1, f);
        fread(&head_idx, 4, 1, f);
        fread(&k, 4, 1, f);
        fread(&d_head, 4, 1, f);
        
        // Read mean and U
        float * mean_host = (float *)malloc(d_head * sizeof(float));
        float * U_host = (float *)malloc(d_head * k * sizeof(float));
        
        fread(mean_host, sizeof(float), d_head, f);
        fread(U_host, sizeof(float), d_head * k, f);
        
        // Copy to GPU
        float * mean_dev, * U_dev;
        cudaMalloc(&mean_dev, d_head * sizeof(float));
        cudaMalloc(&U_dev, d_head * k * sizeof(float));
        
        cudaMemcpy(mean_dev, mean_host, d_head * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(U_dev, U_host, d_head * k * sizeof(float), cudaMemcpyHostToDevice);
        
        // Store in cache params
        cache->subrotq_k.mean = mean_dev;
        cache->subrotq_k.U = U_dev;
        cache->subrotq_k.k = k;
        cache->subrotq_k.d_head = d_head;
        
        free(mean_host);
        free(U_host);
    }
    
    fclose(f);
    cache->use_subrotq = true;
    return true;
}
```

**4.2 Update Ollama to Use Patched llama.cpp**

```bash
# Build custom llama.cpp
cd /tmp/llama.cpp/build
make -j$(nproc)

# Copy binary to Ollama's expected location
sudo systemctl stop ollama
sudo cp /tmp/llama.cpp/build/bin/llama-cli /usr/local/bin/ollama-runner-subrotq
sudo ln -sf /usr/local/bin/ollama-runner-subrotq /usr/local/bin/ollama

# Copy basis file to Ollama model directory
cp gemma4_subrotq_k128_basis.bin ~/.ollama/models/

# Update Ollama config to enable SubRotQ
echo "OLLAMA_SUBROTQ_BASIS=~/.ollama/models/gemma4_subrotq_k128_basis.bin" | sudo tee -a /etc/default/ollama

sudo systemctl start ollama
```

---

### Phase 5: Testing & Validation (Day 3)

**5.1 Functional Tests**

```bash
# Test 1: Basic generation
ollama run gemma4:26b "Count to 100"

# Test 2: Long context (target 40K tokens)
python3 << 'EOF'
import requests

# Generate 40K token prompt (Lorem Ipsum × 10K)
prompt = ("Lorem ipsum dolor sit amet " * 10000)

response = requests.post('http://localhost:11434/api/generate', json={
    'model': 'gemma4:26b',
    'prompt': prompt,
    'stream': False,
    'options': {'num_predict': 50}
})

data = response.json()
print(f"Prompt tokens: {data.get('prompt_eval_count', 'N/A')}")
print(f"Response: {data.get('response', '')[:200]}...")
print(f"Success: {data.get('done', False)}")
EOF
```

**5.2 Quality Tests (Port from exp24/exp25)**

Run PPL and NIAH benchmarks with SubRotQ-enabled Ollama:

```python
# Compare:
# - Baseline (8-bit KV, 28K max)
# - SubRotQ (k=128/4-bit K + 8-bit V, 40K max)

# Expected: SubRotQ rel_PPL ~1.02× (from exp24 Qwen3 result)
```

**5.3 Memory Profiling**

```bash
# Monitor GPU memory during long-context inference
nvidia-smi dmon -s mu -c 100 &
ollama run gemma4:26b "$(python3 -c 'print("test " * 20000)')"
```

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1. Setup | 2h | llama.cpp builds, CUDA works |
| 2. Core Implementation | 8h | SubRotQ kernels + llama.cpp hooks |
| 3. Calibration | 3h | Basis file for Gemma4 26B |
| 4. Integration | 4h | Ollama uses patched llama.cpp |
| 5. Testing | 3h | Quality + memory validation |
| **Total** | **20h** | **~2.5 days** |

---

## Success Criteria

✅ **Functional:**
- Ollama loads Gemma4 26B with SubRotQ compression
- Generates coherent text at 40K+ context length
- No crashes or OOMs

✅ **Performance:**
- Max context: ≥40K tokens on single GPU (GPU1)
- GPU0 remains free (<1GB usage)
- Latency overhead: ≤1.6× (acceptable for prototype)

✅ **Quality:**
- WikiText-2 PPL relative: <1.05× (better than exp24's 0.98× would be a bonus)
- NIAH accuracy: ≥95% at 32K context

✅ **Memory:**
- GPU1 VRAM: ≤24GB at 40K context
- Basis storage: <100MB

---

## Risks & Mitigation

**Risk 1: CUDA kernel bugs**
- Mitigation: Write CPU reference implementation first, validate numerically

**Risk 2: llama.cpp API changes**
- Mitigation: Pin to stable release tag (e.g., `b4313` from Dec 2024)

**Risk 3: Ollama binary replacement breaks**
- Mitigation: Keep backup of original binary, test with small model first

**Risk 4: 40K context still OOMs**
- Fallback: Target 38K context with extra safety margin

**Risk 5: Quality degradation worse than expected**
- Mitigation: If PPL >1.10×, use k=136 or 144 (slight rank increase)

---

## Files to Create

**Code:**
- `/tmp/llama.cpp/ggml-subrotq.h` — Header
- `/tmp/llama.cpp/ggml-subrotq.cu` — CUDA kernels
- `/tmp/llama.cpp/patches/subrotq-integration.patch` — llama.cpp modifications
- `/home/petty/pruning-research/kv-subspace/calibrate_subrotq.py` — Calibration script
- `/home/petty/pruning-research/kv-subspace/export_basis_to_gguf.py` — Basis export
- `/home/petty/pruning-research/kv-subspace/test_subrotq_quality.py` — Validation

**Data:**
- `gemma4_subrotq_k128_basis.pkl` — Python basis (intermediate)
- `gemma4_subrotq_k128_basis.bin` — Binary basis for llama.cpp (~45MB)

**Docs:**
- `SUBROTQ_IMPLEMENTATION_LOG.md` — Daily progress log
- `SUBROTQ_USAGE.md` — How to use SubRotQ-enabled Ollama

---

## Next Steps

1. **Review this plan** — Any concerns or adjustments?
2. **Start Phase 1** — Clone llama.cpp, verify baseline build
3. **Implement incrementally** — Test each phase before moving on

Ready to start? I can generate the first batch of code (Phase 1 + 2.1 scaffolding).
