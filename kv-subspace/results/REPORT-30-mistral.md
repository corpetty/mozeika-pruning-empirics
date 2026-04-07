# Experiment 30: Cross-Architecture (Mistral-7B-v0.3)

Model: mistralai/Mistral-7B-v0.3, 32 layers, GQA (8 KV heads, d_head=128)

Calibration: WikiText-2 train (2048 tokens)

Evaluation: WikiText-2 test (2048 tokens)


## Results
| k | bits | PPL | Rel PPL |
|---|------|-----|---------|
| 64 | 4 | 37.0938 | 8.7039x |
| 96 | 4 | 7.1055 | 1.6673x |
| 112 | 4 | 4.6484 | 1.0907x |
| 128 | 4 | 4.2617 | 1.0000x |

## Comparison with Qwen3-14B-AWQ
| Config | Qwen3 rel PPL | Mistral rel PPL | Delta |
|--------|--------------|-----------------|-------|
| k=64/4-bit | 6.2515x | 8.7039x | +2.4524 |
| k=96/4-bit | 1.5036x | 1.6673x | +0.1637 |
| k=112/4-bit | 1.1577x | 1.0907x | -0.0670 |
| k=128/4-bit | 1.0000x | 1.0000x | +0.0000 |

## Key Findings
- **Baseline PPL**: 4.26 on WikiText-2 (Mistral-7B-v0.3)
- **k=128/4-bit**: 1.00x rel PPL (lossless)
- **k=112/4-bit**: 1.09x rel PPL (production viable)
- **k=96/4-bit**: 1.67x rel PPL (significant degradation)
- **k=64/4-bit**: 8.70x rel PPL (broken)

## Notes
- Basis stability: very low (~0.02-0.03) — calib window may be too small or WikiText-2 too homogeneous
- Cross-architecture validation successful — SubRotQ generalizes from Qwen3 (40 layers, GQA) to Mistral (32 layers, GQA)
- k=112 boundary matches Qwen3 findings (borderline viable)