# Experiment 26: Latency Profiling — Hook Overhead Breakdown

## Config
- Model: Qwen/Qwen3-14B-AWQ
- Context length: 4096
- Decode steps: 64
- Compression: k=128, 4-bit SubRotQ

## Results

| Variant | tok/s | Slowdown vs baseline |
|---------|-------|---------------------|
| Baseline (no hooks) | 12.3 | 1.0× |
| Noop hooks (dispatch only) | 12.3 | 1.0× |
| CPU copy only (no compute) | 11.7 | 1.1× |
| Numpy compression (current) | 7.9 | 1.6× |
| Torch GPU compression (no CPU copy) | 6.0 | 2.1× |

## Overhead Attribution

| Component | Slowdown | Notes |
|-----------|----------|-------|
| Python hook dispatch | 1.0× | Empty hook body |
| GPU↔CPU transfer | 1.0× | detach().cpu() + .to(gpu) |
| Numpy PCA+quant compute | 1.5× | Actual compression |
| **Total (current)** | **1.6×** | Python hooks over numpy |
| Torch GPU (no copy) | 2.1× | Upper bound for fused kernel |

## Key Finding

The paper's claim of "1.2× overhead for a fused kernel" is **not supported**.

- Current implementation: **1.6×** slowdown (Python hooks + numpy + CPU copy)
- Primary bottleneck: GPU↔CPU copy (1.0×) + hook dispatch (1.0×)
- A GPU-native torch implementation (no CPU copy): **2.1×**
- A true fused CUDA kernel could potentially approach 1.1–1.5× but would require
  implementing quantization kernels in CUDA — outside scope of this work

**Honest paper claim:** "Our prototype incurs 2× latency overhead due to
Python hook dispatch and GPU↔CPU transfers. A production CUDA implementation
eliminating these would reduce overhead to an estimated 2.1×, but
we leave kernel implementation as future work."
