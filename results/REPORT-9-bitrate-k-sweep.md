# Experiment 9: Bitrate and Subspace Dimension Sweep

## Setup

- Model: Qwen3-14B-AWQ (40 layers, d_head=128)
- K compression: subspace (k < 128) or full-dim (k=128) PolarQuant
- V compression: full-dim PolarQuant at same n_bits as K
- 3 evaluation passages (scientific, historical, philosophical)
- Sequence length: 512 tokens
- Baseline mean PPL: 2.58

## PPL Heatmap (Mean PPL)

| k \ n_bits | 4 | 6 | 8 | 16 |
|------------|--------|--------|--------|--------|
| k=64 | 8.25 (3.19x) | 6.51 (2.52x) | 6.43 (2.49x) | 6.41 (2.48x) |
| k=96 | 3.25 (1.26x) | 3.04 (1.18x) ** | 3.03 (1.17x) ** | 3.03 (1.17x) ** |
| k=112 | 2.95 (1.14x) ** | 2.82 (1.09x) ** | 2.81 (1.09x) ** | 2.80 (1.09x) ** |
| k=128 | 2.72 (1.05x) ** | 2.60 (1.01x) ** | 2.59 (1.00x) ** | — |

**Bold** = within 20% of baseline.

## Compression Ratio Table

| k \ n_bits | 4 | 6 | 8 | 16 |
|------------|--------|--------|--------|--------|
| k=64 | 5.33x | 3.56x | 2.67x | 1.33x |
| k=96 | 4.57x | 3.05x | 2.29x | 1.14x |
| k=112 | 4.27x | 2.84x | 2.13x | 1.07x |
| k=128 | 4.00x | 2.67x | 2.00x | 1.00x |

## Configs Within 20% PPL Threshold (rel_ppl ≤ 1.20)

| Config | k | n_bits | Mean PPL | Rel PPL | CR |
|--------|---|--------|----------|---------|-----|
| k112_4bit | 112 | 4 | 2.95 | 1.14x | 4.27x |
| k128_4bit | 128 | 4 | 2.72 | 1.05x | 4.00x |
| k96_6bit | 96 | 6 | 3.04 | 1.18x | 3.05x |
| k112_6bit | 112 | 6 | 2.82 | 1.09x | 2.84x |
| k128_6bit | 128 | 6 | 2.60 | 1.01x | 2.67x |
| k96_8bit | 96 | 8 | 3.03 | 1.17x | 2.29x |
| k112_8bit | 112 | 8 | 2.81 | 1.09x | 2.13x |
| k128_8bit | 128 | 8 | 2.59 | 1.00x | 2.00x |
| k96_16bit | 96 | 16 | 3.03 | 1.17x | 1.14x |
| k112_16bit | 112 | 16 | 2.80 | 1.09x | 1.07x |

## Configs Within 50% PPL Threshold (rel_ppl ≤ 1.50)

| Config | k | n_bits | Mean PPL | Rel PPL | CR |
|--------|---|--------|----------|---------|-----|
| k96_4bit | 96 | 4 | 3.25 | 1.26x | 4.57x |
| k112_4bit | 112 | 4 | 2.95 | 1.14x | 4.27x |
| k128_4bit | 128 | 4 | 2.72 | 1.05x | 4.00x |
| k96_6bit | 96 | 6 | 3.04 | 1.18x | 3.05x |
| k112_6bit | 112 | 6 | 2.82 | 1.09x | 2.84x |
| k128_6bit | 128 | 6 | 2.60 | 1.01x | 2.67x |
| k96_8bit | 96 | 8 | 3.03 | 1.17x | 2.29x |
| k112_8bit | 112 | 8 | 2.81 | 1.09x | 2.13x |
| k128_8bit | 128 | 8 | 2.59 | 1.00x | 2.00x |
| k96_16bit | 96 | 16 | 3.03 | 1.17x | 1.14x |
| k112_16bit | 112 | 16 | 2.80 | 1.09x | 1.07x |

## PPL vs Compression Pareto Frontier

| Config | k | n_bits | Mean PPL | Rel PPL | CR | Pareto? |
|--------|---|--------|----------|---------|-----|---------|
| k64_4bit | 64 | 4 | 8.25 | 3.19x | 5.33x | YES |
| k96_4bit | 96 | 4 | 3.25 | 1.26x | 4.57x | YES |
| k112_4bit | 112 | 4 | 2.95 | 1.14x | 4.27x | YES |
| k128_4bit | 128 | 4 | 2.72 | 1.05x | 4.00x | YES |
| k64_6bit | 64 | 6 | 6.51 | 2.52x | 3.56x | no |
| k96_6bit | 96 | 6 | 3.04 | 1.18x | 3.05x | no |
| k112_6bit | 112 | 6 | 2.82 | 1.09x | 2.84x | no |
| k64_8bit | 64 | 8 | 6.43 | 2.49x | 2.67x | no |
| k128_6bit | 128 | 6 | 2.60 | 1.01x | 2.67x | YES |
| k96_8bit | 96 | 8 | 3.03 | 1.17x | 2.29x | no |
| k112_8bit | 112 | 8 | 2.81 | 1.09x | 2.13x | no |
| k128_8bit | 128 | 8 | 2.59 | 1.00x | 2.00x | YES |
| k64_16bit | 64 | 16 | 6.41 | 2.48x | 1.33x | no |
| k96_16bit | 96 | 16 | 3.03 | 1.17x | 1.14x | no |
| k112_16bit | 112 | 16 | 2.80 | 1.09x | 1.07x | no |

## Equal Bit-Budget Comparison: Truncation vs Quantization

At roughly equal total bits per KV pair, which wins on PPL?

| Config | K bits | V bits | Total bits | Mean PPL | Rel PPL |
|--------|--------|--------|------------|----------|---------|
| k64_8bit | 512 | 1024 | 1536 | 6.43 | 2.49x |
| k96_6bit | 576 | 768 | 1344 | 3.04 | 1.18x |
| k112_4bit | 448 | 512 | 960 | 2.95 | 1.14x |
| k128_4bit | 512 | 512 | 1024 | 2.72 | 1.05x |

At ~equal bit budget, k=128/4bit < k=64/8bit → **more dimensions wins** (truncation error dominates).

## Isolating Truncation vs Quantization Error

| Config | Error source | Mean PPL | Rel PPL |
|--------|-------------|----------|---------|
| k128_4bit | Pure quantization (no truncation) | 2.72 | 1.05x |
| k64_16bit | Pure truncation (16-bit ≈ lossless quant) | 6.41 | 2.48x |

**Truncation error dominates**: pure truncation (k64/16bit, 6.41) is worse than pure quantization (k128/4bit, 2.72). Retaining all 128 dims matters more than having high bit precision.

## Recommendation

**Minimum viable config (best CR within ≤20% PPL):** k=112, 4-bit
- Mean PPL: 2.95 (1.14x baseline)
- Compression ratio: 4.27x
- K storage: 112×4 = 448 bits, V storage: 128×4 = 512 bits
- Total: 960 bits vs 4096 bits FP16
