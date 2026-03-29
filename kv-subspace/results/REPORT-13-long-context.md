# Experiment 13: Long-Context KV Cache Compression Scaling

## Setup

- Model: Qwen3-14B-AWQ (40 layers, 8 GQA KV heads, d_head=128)
- Text: War and Peace (Project Gutenberg) — continuous long document
- Calibration: first 2048 tokens of document
- Evaluation: held-out text (offset from calibration to avoid overlap)

## Sub-experiment A: PPL vs Context Length

### Absolute PPL

| Config | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 40960 |
|--------|------|------|------|------|------|------|------|------|
| baseline | 8.875 | 10.000 | 10.117 | 10.102 | 9.602 | 10.000 | 9.978 | 10.116 |
| k128_4bit | 9.805 | 11.094 | 10.984 | 10.750 | 10.102 | 10.625 | 10.618 | 11.049 |
| k112_4bit | 14.898 | 14.695 | 15.578 | 13.992 | 12.914 | 14.219 | 16.501 | 16.941 |
| k96_4bit | 20.406 | 20.969 | 19.281 | 16.328 | 15.891 | 17.141 | 18.227 | 18.747 |
| k64_4bit | 133.000 | 107.750 | 72.625 | 60.656 | 49.812 | 45.875 | 43.581 | 43.119 |

### Relative PPL (compressed / baseline)

| Config | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 40960 |
|--------|------|------|------|------|------|------|------|------|
| k128_4bit | 1.105 | 1.109 | 1.086 | 1.064 | 1.052 | 1.062 | 1.064 | 1.092 |
| k112_4bit | 1.679 | 1.470 | 1.540 | 1.385 | 1.345 | 1.422 | 1.654 | 1.675 |
| k96_4bit | 2.299 | 2.097 | 1.906 | 1.616 | 1.655 | 1.714 | 1.827 | 1.853 |
| k64_4bit | 14.986 | 10.775 | 7.178 | 6.005 | 5.188 | 4.588 | 4.368 | 4.263 |

### Key Finding: Does Relative PPL Drift With Context Length?

- **k128_4bit**: short-ctx rel_PPL=1.105, long-ctx=1.092, delta=-0.013 → **STABLE**
- **k112_4bit**: short-ctx rel_PPL=1.679, long-ctx=1.675, delta=-0.004 → **STABLE**
- **k96_4bit**: short-ctx rel_PPL=2.299, long-ctx=1.853, delta=-0.446 → **IMPROVING**
- **k64_4bit**: short-ctx rel_PPL=14.986, long-ctx=4.263, delta=-10.723 → **IMPROVING**

## Sub-experiment B: Per-Token Loss at 16K Context

Are compression errors uniform across sequence positions, or do they concentrate in later tokens?

Mean loss per decile of sequence (0%=early, 100%=late):

| Config | 0-10% | 10-20% | 20-30% | 30-40% | 40-50% | 50-60% | 60-70% | 70-80% | 80-90% | 90-100% |
|--------|------|------|------|------|------|------|------|------|------|------|
| baseline | 2.286 | 2.337 | 2.247 | 2.331 | 2.101 | 2.202 | 2.404 | 2.276 | 2.407 | 2.441 |
| k112_4bit | 2.765 | 2.619 | 2.522 | 2.600 | 2.428 | 2.517 | 2.760 | 2.660 | 2.871 | 2.808 |
| k64_4bit | 4.540 | 3.932 | 3.810 | 3.796 | 3.522 | 3.662 | 3.800 | 3.757 | 3.711 | 3.723 |

## Sub-experiment C: Calibration Basis Drift

Subspace overlap (cos²θ, 1=identical, 0=orthogonal) between PCA basis fitted at document start vs later positions:

| Compare Position | K overlap | V overlap |
|-----------------|-----------|-----------|
| early vs mid_early | 0.831 | 0.716 |
| early vs mid_late | 0.827 | 0.705 |
| early vs late | 0.825 | 0.702 |

A drop in overlap indicates the PCA basis fitted on early tokens is less representative of late-document KV distributions.

## Conclusions and Recommendations

See per-section findings above. Key question: does compression quality hold at long context, or is there a context length beyond which the calibrated basis and/or cascade effects cause meaningful degradation?

---
*Experiment 13 — part of the KV cache subspace compression research.*
*Repo: https://github.com/corpetty/mozeika-pruning-empirics*
