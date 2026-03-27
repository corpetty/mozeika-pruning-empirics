# Experiment 11: Cross-Model Validation — k/d_head Generalization

## Question

Is the k/d_head >= 0.875 rule (found on Qwen3-14B-AWQ, d_head=128) a general
principle or model-specific?

## Models Tested

| Model | Layers | KV Heads | d_head | AWQ |
|-------|--------|----------|--------|-----|
| Qwen3-1.7B | 28 | 8 | 128 | No |
| Qwen3-32B-AWQ | 64 | 8 | 128 | Yes |
| Qwen3-14B-AWQ (Exp 9) | 40 | 8 | 128 | Yes |

## Setup

- K compression: subspace PCA + PolarQuant at 4-bit
- V compression: full-dim PolarQuant at 4-bit
- k/d_head fractions tested: 0.50, 0.75, 0.875, 0.9375, 1.0
- Calibration: Project Gutenberg text, in-memory PCA (2048 tokens for 1.7B; 512 for 32B)
- Evaluation: 3 passages (scientific, historical, philosophical), 512 tokens each

## 1. PPL vs k/d_head (Side by Side)

| k/d_head | Qwen3-1.7B (base=4.59) | Qwen3-32B-AWQ (base=2.27) | 14B-AWQ (Exp 9) |
|----------|---------------------------|---------------------------|-----------------|
| 0.5000 | k=64: 54.42 (11.85x, CR=5.33x) | k=64: 4.59 (2.02x, CR=5.33x) | — |
| 0.7500 | k=96: 8.29 (1.80x, CR=4.57x) | k=96: 2.48 (1.09x, CR=4.57x) ** | — |
| 0.8750 | k=112: 5.98 (1.30x, CR=4.27x) | k=112: 2.41 (1.06x, CR=4.27x) ** | k=112: 1.14x |
| 0.9375 | k=120: 5.82 (1.27x, CR=4.13x) | k=120: 2.38 (1.05x, CR=4.13x) ** | — |
| 1.0000 | k=128: 5.18 (1.13x, CR=4.00x) ** | k=128: 2.36 (1.04x, CR=4.00x) ** | k=128: 1.05x |

**Bold** = within 20% PPL degradation threshold.

## 2. Does k/d_head >= 0.875 Hold?

### Qwen3-1.7B (d_head=128)

- At k/d_head=0.875 (k=112): rel_ppl = 1.30x → within 20%? **NO**
- Smallest k/d_head within 20%: 1.0000 (k=128)

### Qwen3-32B-AWQ (d_head=128)

- At k/d_head=0.875 (k=112): rel_ppl = 1.06x → within 20%? **YES**
- Smallest k/d_head within 20%: 0.7500 (k=96)

### 14B-AWQ (reference from Exp 9)

- At k/d_head=0.875 (k=112): rel_ppl = 1.14x → within 20%? **YES**
- At k/d_head=1.0 (k=128): rel_ppl = 1.05x

## 3. PPL vs Compression Pareto

### Qwen3-1.7B

| k/d_head | k | Mean PPL | Rel PPL | CR | Pareto? |
|----------|---|----------|---------|-----|---------|
| 0.5000 | 64 | 54.42 | 11.85x | 5.33x | YES |
| 0.7500 | 96 | 8.29 | 1.80x | 4.57x | YES |
| 0.8750 | 112 | 5.98 | 1.30x | 4.27x | YES |
| 0.9375 | 120 | 5.82 | 1.27x | 4.13x | YES |
| 1.0000 | 128 | 5.18 | 1.13x | 4.00x | YES |

### Qwen3-32B-AWQ

| k/d_head | k | Mean PPL | Rel PPL | CR | Pareto? |
|----------|---|----------|---------|-----|---------|
| 0.5000 | 64 | 4.59 | 2.02x | 5.33x | YES |
| 0.7500 | 96 | 2.48 | 1.09x | 4.57x | YES |
| 0.8750 | 112 | 2.41 | 1.06x | 4.27x | YES |
| 0.9375 | 120 | 2.38 | 1.05x | 4.13x | YES |
| 1.0000 | 128 | 2.36 | 1.04x | 4.00x | YES |

## 4. Cross-Model Comparison at k/d_head=0.875

| Model | d_head | k | Rel PPL | CR |
|-------|--------|---|---------|-----|
| Qwen3-14B-AWQ | 128 | 112 | 1.14x | 4.27x |
| Qwen3-1.7B | 128 | 112 | 1.30x | 4.27x |
| Qwen3-32B-AWQ | 128 | 112 | 1.06x | 4.27x |

### Pattern: Model Size vs Compression Tolerance

At k/d_head=0.875, 4-bit:

- Qwen3-1.7B: 1.30x
- Qwen3-32B-AWQ: 1.06x
- Qwen3-14B-AWQ: 1.14x

Trend: larger models tolerate compression better (lower rel_ppl).

## 5. Conclusion

**k/d_head >= 0.875 does NOT universally hold.** Results vary by model:

- Qwen3-1.7B: exceeds 20% (1.30x)
- Qwen3-32B-AWQ: within 20%

The threshold may need to be adjusted per model size — smaller models
are more sensitive to subspace truncation and may need k/d_head closer to 1.0.
