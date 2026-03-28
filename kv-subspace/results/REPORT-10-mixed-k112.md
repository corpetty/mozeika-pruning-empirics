# Experiment 10: Mixed Layerwise Policy with k=112

## Context

Exp 8 showed that mixed layerwise policies with k=64/4-bit still caused
1.75-3x PPL degradation. The cascade hypothesis was confirmed but the best
policy (protect L0-19) only achieved 1.68x compression at 1.75x PPL.

Exp 9 revealed the root cause: **truncation error dominates**. k=64 throws
away too much signal. k=112/4-bit achieves 4.27x compression with only
1.14x PPL degradation (uniform across all layers).

**Key question**: Does the cascade problem from Exp 8 disappear at k=112?

## Setup

- Model: Qwen3-14B-AWQ (40 layers, d_head=128)
- 3 evaluation passages (scientific, historical, philosophical)
- Sequence length: 512 tokens
- Compression via forward hooks on k_proj/v_proj outputs
- PCA bases from calibration data (results/kvs.npz)

## Policies Tested

| Policy | Description | Layers Compressed |
|--------|-------------|-------------------|
| baseline | No compression | 0/40 |
| k96_uniform | k=96/4-bit K + full-dim 4-bit V, all layers | 40/40 |
| k112_uniform | k=112/4-bit K + full-dim 4-bit V, all layers | 40/40 |
| k112_protect10 | Skip L0-9, k=112/4-bit K + full-dim 4-bit V on L10-39 | 30/40 |
| k112_protect20 | Skip L0-19, k=112/4-bit K + full-dim 4-bit V on L20-39 | 20/40 |
| k112_graduated | Skip L0-9, K-only L10-19, full KV L20-39 | 30/40 (graduated) |
| k128_uniform | Full-dim 4-bit K + V, all layers (no truncation) | 40/40 |
| hybrid_128_early_112_late | k=128/4-bit L0-19, k=112/4-bit L20-39 | 40/40 |

## Results

| Policy | P0 | P1 | P2 | Mean PPL | Rel PPL | Compression Ratio |
|--------|-----|-----|-----|----------|---------|-------------------|
| baseline | 2.09 | 2.93 | 2.73 | 2.58 | 1.00x | 1.00x |
| k96_uniform | 2.44 | 3.92 | 3.39 | 3.25 | 1.26x | 4.57x |
| k112_uniform | 2.37 | 3.42 | 3.05 | 2.95 | 1.14x | 4.27x |
| k112_protect10 | 2.32 | 3.48 | 3.08 | 2.96 | 1.15x | 2.35x |
| k112_protect20 | 2.27 | 3.28 | 3.07 | 2.87 | 1.11x | 1.62x |
| k112_graduated | 2.34 | 3.36 | 3.08 | 2.93 | 1.13x | 1.92x |
| k128_uniform | 2.21 | 3.09 | 2.87 | 2.72 | 1.05x | 4.00x |
| hybrid_128_early_112_late | 2.28 | 3.19 | 3.02 | 2.83 | 1.10x | 4.13x |

## Q1: Does k=112 Uniform Already Solve the Cascade?

Exp 8 (k=64): uniform_kv_optimal = 3.19x PPL — severe cascade degradation.
Exp 10 (k=112): k112_uniform = 1.14x PPL

**Yes** — k=112 uniform is already within 20% of baseline (1.14x).
The cascade effect that plagued k=64 is largely eliminated by retaining
more dimensions. The error per layer is small enough that 40-layer
accumulation stays manageable.

## Q2: Do Mixed Policies Improve Over k=112 Uniform?

- k112_uniform: 2.95 PPL (gap from baseline: 0.36)
- k112_protect10: 2.96 PPL (gap: 0.38)
- k112_protect20: 2.87 PPL (gap: 0.29)

Protecting L0-9 recovers -3.4% of the PPL gap.
Protecting L0-19 recovers 20.6% of the PPL gap.

But the compression cost is significant:
- k112_uniform: 4.27x CR
- k112_protect10: 2.35x CR (45% less compression)
- k112_protect20: 1.62x CR (62% less compression)

## Q3: PPL Gain Per Protected Layer at k=112

- Protecting 10 layers: -0.001 PPL / layer (-0.05% rel PPL / layer)
- Protecting 20 layers: 0.004 PPL / layer (0.14% rel PPL / layer)

For comparison, Exp 8 at k=64 showed ~3%/layer gain. At k=112 we expect
much smaller per-layer gains since the per-layer error is already small.

## Pareto Frontier: PPL vs Compression Ratio

| Policy | Mean PPL | Rel PPL | CR | Pareto? |
|--------|----------|---------|-----|---------|
| k96_uniform | 3.25 | 1.26x | 4.57x | YES |
| k112_uniform | 2.95 | 1.14x | 4.27x | YES |
| hybrid_128_early_112_late | 2.83 | 1.10x | 4.13x | YES |
| k128_uniform | 2.72 | 1.05x | 4.00x | YES |
| k112_protect10 | 2.96 | 1.15x | 2.35x | no |
| k112_graduated | 2.93 | 1.13x | 1.92x | no |
| k112_protect20 | 2.87 | 1.11x | 1.62x | no |

## Final Recommended Config (Across All 10 Experiments)

### Within 20% PPL threshold:

| Policy | Mean PPL | Rel PPL | CR |
|--------|----------|---------|-----|
| k112_uniform | 2.95 | 1.14x | 4.27x | **<-- best CR**
| hybrid_128_early_112_late | 2.83 | 1.10x | 4.13x |
| k128_uniform | 2.72 | 1.05x | 4.00x | **<-- best PPL**
| k112_protect10 | 2.96 | 1.15x | 2.35x |
| k112_graduated | 2.93 | 1.13x | 1.92x |
| k112_protect20 | 2.87 | 1.11x | 1.62x |

### Recommendation:

**Primary recommendation: k128_uniform**
- Rel PPL: 1.05x
- Compression ratio: 4.00x
- Efficiency (rel_ppl/CR): 0.2635

**Tiered options for different use cases:**

1. **Maximum compression** (best CR within 20% PPL): k=112/4-bit uniform
   - 1.14x PPL, 4.27x CR
2. **Minimum PPL impact** (closest to baseline): k=128/4-bit uniform
   - 1.05x PPL, 4.00x CR
3. **Balanced**: hybrid k=128 early / k=112 late
   - 1.10x PPL, 4.13x CR

**Key insight from 10 experiments:** Truncation error (not quantization)
is the dominant factor. The cascade effect that makes k=64 impractical
(3.19x PPL) largely disappears at k=112 (1.14x PPL) because per-layer
error is small enough that 40-layer accumulation stays within budget.
Mixed policies offer marginal PPL gains but at significant compression cost.
