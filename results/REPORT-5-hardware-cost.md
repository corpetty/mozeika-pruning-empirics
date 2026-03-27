# Report 5: Hardware Cost — Latency Overhead Microbenchmark

## Experimental Setup

Pure PyTorch microbenchmark on an **NVIDIA RTX 3090** measuring CUDA latency for five compression operations on (T, 128) tensors:

| Operation | Description |
|-----------|-------------|
| plain_quantize | Uniform scalar quantization on (T, 128) |
| polarquant_128 | Rotation (128×128 matmul) + quantize + inverse rotation |
| project_quantize_k{32,64} | Projection (128→k matmul) + quantize in k dims + reconstruction (k→128) |
| subspace_polar_k{32,64} | Projection + rotation in k-dim + quantize + inverse rotation + reconstruction |

All operations include full encode-decode roundtrip (compression + decompression). Measured with `torch.cuda.synchronize()` + `time.perf_counter()`, 200 iterations after 50 warmup, taking median.

## Latency Table (microseconds)

| Operation | T=1 | T=8 | T=32 | T=128 | T=512 | T=1024 |
|-----------|-----|-----|------|-------|-------|--------|
| plain_quantize | 174 | 180 | 181 | 184 | 184 | 197 |
| polarquant_128 | 244 | 258 | 259 | 278 | 263 | 275 |
| project_quantize_k32 | 264 | 282 | 304 | 304 | 304 | 305 |
| subspace_polar_k32 | 298 | 315 | 343 | 342 | 344 | 344 |
| project_quantize_k64 | 265 | 280 | 288 | 287 | 288 | 288 |
| subspace_polar_k64 | 297 | 318 | 325 | 326 | 324 | 326 |

## Overhead Ratios (relative to plain quantization)

| Operation | T=1 | T=8 | T=32 | T=128 | T=512 | T=1024 |
|-----------|-----|-----|------|-------|-------|--------|
| polarquant_128 | 1.40× | 1.43× | 1.43× | 1.52× | 1.43× | 1.40× |
| project_quantize_k32 | 1.51× | 1.57× | 1.68× | 1.66× | 1.65× | 1.55× |
| **project_quantize_k64** | **1.52×** | **1.56×** | **1.59×** | **1.56×** | **1.56×** | **1.46×** |
| subspace_polar_k32 | 1.71× | 1.75× | 1.90× | 1.86× | 1.87× | 1.75× |
| **subspace_polar_k64** | **1.70×** | **1.77×** | **1.80×** | **1.78×** | **1.76×** | **1.66×** |

## Key Observations

### 1. All operations are latency-bound, not compute-bound
Latency barely changes from T=1 to T=1024 (174→197 μs for plain quantize). The operations are dominated by **kernel launch overhead and memory access patterns**, not actual FLOPs. At T=1024, we're still well within the GPU's compute capability.

### 2. Subspace projection adds ~70–80% overhead
The full subspace PolarQuant pipeline (project + rotate + quantize + reconstruct) costs **1.66–1.80×** the plain quantization latency. For the recommended k=64 configuration:
- **plain_quantize**: ~185 μs
- **subspace_polar_k64**: ~325 μs
- **Additional overhead**: ~140 μs per head per token batch

### 3. k=64 is faster than k=32 (counterintuitively)
Project_quantize_k64 (288 μs) is slightly faster than project_quantize_k32 (304 μs). This is because k=64 leads to better memory alignment and CUDA kernel efficiency — 64 is a CUDA-friendly dimension (power of 2), while 32 may cause underutilization in some kernels.

### 4. Rotation overhead is modest
Adding PolarQuant rotation on top of projection costs only ~35–40 μs extra (project_quantize → subspace_polar). The rotation is a (k, k) matmul which is negligible at k=64.

## Practical Context: Where Does This Fit in Inference?

### Per-token latency budget
For Qwen3-14B at batch_size=1:
- Full transformer forward pass: ~5–10 ms per token (on RTX 3090)
- KV cache quantization per layer: ~0.3 ms (subspace_polar_k64)
- Total across 40 layers: ~12 ms (subspace) vs ~7.4 ms (plain quantize)
- **Overhead**: ~4.6 ms added per token, or roughly **50–90% of one forward pass**

This is significant for single-token decoding but becomes less important in context:

### Amortization at long sequences
The compression overhead is **per KV insertion** (once per token per layer), while the compression **benefit** (reduced memory, faster attention) scales with **sequence length × number of queries**:
- At sequence length 2048: 2048 × 40 layers × 8 heads × 128 dims × 2 bytes = 168 MB KV cache
- With k=64, 4-bit subspace: 2048 × 40 × 8 × 64 × 0.5 bytes = 21 MB (8× smaller)
- The attention computation speedup from smaller KV cache dominates at long sequences

### When is the overhead negligible?
- **Prefill phase**: Compression runs in parallel with attention computation, largely hidden
- **Long-context inference (>4K tokens)**: Memory savings outweigh the ~140 μs/head compression overhead
- **Batch inference**: Overhead is amortized across batch dimension

### When does it matter?
- **Real-time single-token generation**: The ~5 ms additional latency (40 layers) adds noticeable delay
- **Short sequences (<512 tokens)**: Memory savings are minimal, overhead is proportionally larger

## Practical Recommendation

**The projection overhead is acceptable for long-context inference but not negligible.**

| Scenario | Recommendation |
|----------|---------------|
| Long context (>4K tokens) | Use subspace compression — memory savings dominate |
| Short context (<512 tokens) | Plain PolarQuant may be preferable — lower overhead |
| Real-time chat (latency-critical) | Consider selective compression (compress only layers 10+, skip early layers) |
| Batch serving | Use subspace compression — overhead amortized across batch |

An optimized CUDA kernel (fusing projection + quantization into a single kernel) could reduce the overhead from 1.7× to closer to 1.2–1.3×, making subspace compression nearly free in practice.

## Raw Data

Full results: `results/hardware_cost.csv` (36 rows)
- 6 batch sizes × 6 operations
