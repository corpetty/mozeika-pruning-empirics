# Experiment 25: Needle-in-a-Haystack (Robust, n=15 per cell)

- Model: Qwen/Qwen3-14B-AWQ
- 15 needles × 5 depths × 4 ctx_lengths × 3 configs
- Calibration: WikiText-2 train (not W&P — no memorization)

## Accuracy by Config × Context Length (% correct, 95% Wilson CI)

| Config | ctx=4096 | ctx=8192 | ctx=16384 | ctx=32768 | Overall |
|--------|----------|----------|-----------|-----------|---------|
| baseline | 100% (75/75) [95–100%] | 100% (75/75) [95–100%] | 100% (75/75) [95–100%] | 100% (75/75) [95–100%] | 100% (300/300) [99–100%] |
| k128_4bit | 99% (74/75) [93–100%] | 100% (75/75) [95–100%] | 100% (75/75) [95–100%] | 100% (75/75) [95–100%] | 100% (299/300) [98–100%] |
| k96_4bit | 99% (74/75) [93–100%] | 100% (75/75) [95–100%] | 100% (75/75) [95–100%] | 97% (73/75) [91–99%] | 99% (297/300) [97–100%] |

## Accuracy by Config × Depth (% correct, all ctx_lengths pooled)

| Config | 10% | 25% | 50% | 75% | 90% |
|--------|-----|-----|-----|-----|-----|
| baseline | 100% [94–100%] | 100% [94–100%] | 100% [94–100%] | 100% [94–100%] | 100% [94–100%] |
| k128_4bit | 100% [94–100%] | 100% [94–100%] | 98% [91–100%] | 100% [94–100%] | 100% [94–100%] |
| k96_4bit | 97% [89–99%] | 100% [94–100%] | 98% [91–100%] | 100% [94–100%] | 100% [94–100%] |

## Notes
- All accuracy estimates reported with 95% Wilson confidence intervals
- n=15 per (config, depth, ctx_len) cell
- Prior exp15 had n=3/cell — single miss changed accuracy by 33pp
- W&P-based calibration replaced with WikiText-2 train split