# Experiment 27: Downstream Task Accuracy

- Model: Qwen/Qwen3-14B-AWQ
- Tasks: arc_challenge, hellaswag, arc_easy, winogrande
- Samples/task: 300 (MC tasks only)
- Calibration: WikiText-2 train split, K-only SubRotQ compression

## Accuracy Summary

| Config | ARC-C | HellaSwag | ARC-Easy | WinoGrande |
|--------|-------|-----------|----------|-----------|
| baseline | 0.677 | 0.557 | 0.787 | 0.777 |
| k128_4bit | 0.647 (-0.030) | 0.553 (-0.003) | 0.790 (+0.003) | 0.753 (-0.023) |
| k112_4bit | 0.607 (-0.070) | 0.520 (-0.037) | 0.747 (-0.040) | 0.707 (-0.070) |
| k96_4bit | 0.507 (-0.170) | — | — | — |

## Confidence Margin Summary

(mean log-prob margin: correct choice log-prob minus max distractor log-prob)

| Config | ARC-C margin | HSwag margin | ARC-Easy margin | WinoGrande margin |
|--------|-------------|-------------|----------------|-----------------|
| baseline | 2.399 | 0.168 | 4.814 | -1.839 |
| k128_4bit | 2.022 | -0.230 | 4.797 | -1.762 |
| k112_4bit | 0.635 | -1.621 | 4.176 | -1.486 |
| k96_4bit | -0.490 | — | — | — |

## Key Findings
- Δ shown as (compressed − baseline); negative = degraded
- Margin = log-prob correct − log-prob best distractor; collapse here indicates fragility even when accuracy holds

## Diagnostics
- Variance explained: k96=0.982, k112=0.994, k128=1.000
- Basis stability (calib first/second half cosine sim): mean=0.396, min=0.335
- Truncation vs quantization error split: truncation=0.000