# Experiment 12: Cross-Architecture Validation — Mistral-7B + Phi-4

## Question

Is the k/d_head >= 0.875 rule architecture-dependent or universal?
Does it hold for Mistral and Phi3 architectures (not just Qwen3)?

## Models Tested

| Model | Architecture | Params | Layers | KV Heads | d_head | Quantization |
|-------|-------------|--------|--------|----------|--------|-------------|
| Mistral-7B-v0.3 | Mistral | 7B | 32 | 8 | 128 | BF16 |
| Phi-4-AWQ | Phi3 | 14B | 40 | 10 | 128 | AWQ |
| Qwen3-14B-AWQ (ref) | Qwen3 | 14B | 40 | 8 | 128 | AWQ |

## Setup

- K compression: subspace PCA + PolarQuant at 4-bit
- V compression: full-dim PolarQuant at 4-bit
- k/d_head fractions tested: 0.50, 0.75, 0.875, 0.9375, 1.0
- Calibration: Project Gutenberg text, in-memory PCA (512 tokens)
- Evaluation: 3 passages (scientific, historical, philosophical), 512 tokens each

## 1. PPL vs k/d_head (Side by Side)

| k/d_head | Mistral-7B-v0.3 (base=2.68) | Phi-4-AWQ (base=2.72) | 14B-AWQ (ref) |
|----------|---------------------------|---------------------------|----------------|
| 0.5000 | k=64: 2.99 (1.12x) ** | k=64: 3.20 (1.18x) ** | k=64: 8.25 (3.19x) |
| 0.7500 | k=96: 2.92 (1.09x) ** | k=96: 3.05 (1.12x) ** | k=96: 3.25 (1.26x) |
| 0.8750 | k=112: 2.87 (1.07x) ** | k=112: 2.98 (1.10x) ** | k=112: 2.95 (1.14x) |
| 0.9375 | k=120: 2.88 (1.07x) ** | k=120: 2.95 (1.09x) ** | — |
| 1.0000 | k=128: 2.81 (1.05x) ** | k=128: 2.94 (1.08x) ** | k=128: 2.72 (1.05x) |

**Bold** = within 20% PPL degradation threshold.

## 2. Does k/d_head >= 0.875 Hold?

### Mistral-7B-v0.3 (Mistral, 7B, d_head=128)

- At k/d_head=0.875 (k=112): rel_ppl = 1.07x -> within 20%? **YES**
- Smallest k/d_head within 20%: 0.5000 (k=64)

### Phi-4-AWQ (Phi3, 14B, d_head=128)

- At k/d_head=0.875 (k=112): rel_ppl = 1.10x -> within 20%? **YES**
- Smallest k/d_head within 20%: 0.5000 (k=64)

### Qwen3-14B-AWQ (reference from Exp 9)

- At k/d_head=0.875 (k=112): rel_ppl = 1.14x -> within 20%? **YES**
- At k/d_head=1.0 (k=128): rel_ppl = 1.05x

## 3. Cross-Architecture Comparison at k/d_head=0.875

| Model | Architecture | Params | Rel PPL | CR |
|-------|-------------|--------|---------|-----|
| Qwen3-1.7B | Qwen3 | 1.7B | 1.32x | 4.27x |
| Mistral-7B-v0.3 | Mistral | 7B | 1.07x | 4.27x |
| Qwen3-14B-AWQ | Qwen3 | 14B | 1.14x | 4.27x |
| Phi-4-AWQ | Phi3 | 14B | 1.10x | 4.27x |
| Qwen3-32B-AWQ | Qwen3 | 32B | 1.05x | 4.27x |

## 4. Is the Pattern Model-Size-Dependent, Architecture-Dependent, or Both?

- **Mistral-7B (7B, Mistral arch)**: 1.07x
- **Phi-4 (14B, Phi3 arch)**: 1.10x
- **Qwen3-14B (14B, Qwen3 arch)**: 1.14x
- **Qwen3-1.7B (1.7B, Qwen3 arch)**: 1.32x
- **Qwen3-32B (32B, Qwen3 arch)**: 1.05x

### Size vs Architecture Analysis

**Same size, different arch** (Phi-4 vs Qwen3-14B, both 14B):
  Phi-4 = 1.10x, Qwen3-14B = 1.14x, diff = 0.04

**Same arch, different size** (Qwen3 family):
  1.7B = 1.32x, 14B = 1.14x, 32B = 1.05x

**Conclusion**: At matched size (14B), Phi-4 and Qwen3 show similar compression tolerance.
This suggests the k/d_head rule is **primarily size-dependent**, not architecture-dependent.

## 5. Revised k/d_head Rule

All models at k/d_head=0.875, 4-bit:

| Model | Arch | Params | Rel PPL | Within 20%? |
|-------|------|--------|---------|-------------|
| Qwen3-32B-AWQ | Qwen3 | 32B | 1.05x | YES |
| Mistral-7B-v0.3 | Mistral | 7B | 1.07x | YES |
| Phi-4-AWQ | Phi3 | 14B | 1.10x | YES |
| Qwen3-14B-AWQ | Qwen3 | 14B | 1.14x | YES |
| Qwen3-1.7B | Qwen3 | 1.7B | 1.32x | NO |

**4/5 models pass** the k/d_head >= 0.875 rule at 4-bit.

The rule holds for most models. Exception: Qwen3-1.7B.
For these models, k/d_head >= 0.9375 may be needed.

## 6. Recommended Config per Architecture

| Architecture | Model | Recommended k/d_head | k (d_head=128) | Notes |
|-------------|-------|---------------------|-----------------|-------|
| Mistral | Mistral-7B-v0.3 | 0.5000 | 64 | 1.12x rel PPL |
| Phi3 | Phi-4-AWQ | 0.5000 | 64 | 1.18x rel PPL |
| Qwen3 | Qwen3-14B-AWQ | 0.875 | 112 | 1.14x rel PPL (Exp 9) |
| Qwen3 | Qwen3-32B-AWQ | 0.75 | 96 | 1.09x rel PPL (Exp 11) |
| Qwen3 | Qwen3-1.7B | 0.9375+ | 120+ | 1.32x at 0.875 — needs higher k (Exp 11) |

