# Experiment 21: Llama-3.1 Architecture Validation

**Model:** hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4  
**Architecture:** llama (32 layers, 8 KV heads, d_head=128, no QK-norm)  
**Baseline PPL (ctx=512):** 5.3967

## Motivation

All prior experiments used Qwen3-14B-AWQ as the primary model. Qwen3 applies
QK-norm (RMSNorm on k_proj and q_proj outputs) which may force K into a
low-dimensional manifold while leaving V variance undistorted. Llama-3.1 uses
standard GQA without QK-norm — making it the critical test case for the
cross-architecture V compression hypothesis.

## Sub-exp A: K+V Sweep (k/d_head fractions)

| k | k/d_head | PPL | Rel PPL | CR | Within 20% |
|---|----------|-----|---------|-----|------------|
| 64 | 0.5000 | 102.8805 | 19.0637 | 8.00x | ✗ |
| 96 | 0.7500 | 88.3625 | 16.3735 | 5.33x | ✗ |
| 112 | 0.8750 | 75.6979 | 14.0268 | 4.57x | ✗ |
| 120 | 0.9375 | 36.0486 | 6.6798 | 4.27x | ✗ |
| 128 | 1.0000 | 5.8558 | 1.0851 | 4.00x | ✓ |

**Minimum viable k for Llama-3.1-8B (< 20% PPL): 128**

## Sub-exp B: K-only vs V-only at k=112/4-bit

| Config | PPL | Rel PPL | CR |
|--------|-----|---------|-----|
| K-only (V full precision) | 5.6249 | 1.0423 | 4.57x |
| V-only (K full precision) | 65.5270 | 12.1421 | 4.57x |

## Sub-exp C: V Threshold Scan (QK-norm hypothesis test)

K fixed at k=112/4-bit. Viability: gap vs K-only < 0.05x rel PPL.

| k_V | PPL | Rel PPL | Gap vs K-only | CR | Viable |
|-----|-----|---------|---------------|-----|--------|
| 64 | 173.5407 | 32.1570 | +31.1147 | 8.00x | ✗ |
| 96 | 94.9859 | 17.6009 | +16.5586 | 5.33x | ✗ |
| 112 | 75.6979 | 14.0268 | +12.9845 | 4.57x | ✗ |
| 120 | 38.6812 | 7.1676 | +6.1253 | 4.27x | ✗ |
| 124 | 18.6359 | 3.4532 | +2.4109 | 4.13x | ✗ |
| 128 | 6.0398 | 1.1192 | +0.0769 | 4.00x | ✗ |

**Minimum viable k_V for Llama-3.1-8B:** none (V not viable)

## Cross-Architecture Comparison

| Model | Arch | Params | Min k for <20% PPL | Rel PPL at k=112 |
|-------|------|--------|-------------------|------------------|
| Qwen3-1.7B | Qwen3 (QK-norm) | 1.7B | 128 | 1.32x |
| Mistral-7B-v0.3 | Mistral (no QK-norm) | 7B | 64 | 1.07x |
| Qwen3-14B-AWQ | Qwen3 (QK-norm) | 14B | 112 | 1.14x |
| Phi-4-AWQ | Phi3 (no QK-norm) | 14B | 64 | 1.10x |
| Llama-3.1-8B | Llama (no QK-norm) | 8B | 128 | (see above) |

## QK-norm Hypothesis Assessment

Qwen3 applies QK-norm (RMSNorm after k_proj/q_proj) but not V-norm. Llama-3.1,
Mistral-7B, and Phi-3 do not use QK-norm. The hypothesis: QK-norm forces K into
a low-dimensional manifold while V retains full variance, explaining why V
compression fails for Qwen3 but may succeed for other architectures.

If Llama-3.1 V compression is viable at k<128, this supports the QK-norm hypothesis.
If not, V compression is universally hard, and the mechanism is something else
(possibly GQA itself, since all three non-Qwen models also use GQA).
