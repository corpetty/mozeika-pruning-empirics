# Experiment 17: Cross-Domain Calibration Robustness

- Model: Qwen3-14B-AWQ
- Calibration tokens: 1024 per domain
- Eval context: 2048 tokens
- Domains: ['fiction', 'code', 'news', 'dialogue']

## PPL matrix — baseline

| calib↓ / eval→ | fiction | code | news | dialogue |
|---|---|---|---|---|
| fiction | 13.250 | 1.283 | 2.532 | 2.860 |
| code | 13.250 | 1.283 | 2.532 | 2.860 |
| news | 13.250 | 1.283 | 2.532 | 2.860 |
| dialogue | 13.250 | 1.283 | 2.532 | 2.860 |
| universal | 13.250 | 1.283 | 2.532 | 2.860 |

## PPL matrix — k128_4bit

| calib↓ / eval→ | fiction | code | news | dialogue |
|---|---|---|---|---|
| fiction | 13.694 | 1.317 | 2.506 | 2.972 |
| code | 13.341 | 1.333 | 2.565 | 2.910 |
| news | 13.581 | 1.305 | 2.609 | 2.933 |
| dialogue | 13.670 | 1.330 | 2.607 | 2.897 |
| universal | 13.694 | 1.317 | 2.506 | 2.972 |

## PPL matrix — k96_4bit

| calib↓ / eval→ | fiction | code | news | dialogue |
|---|---|---|---|---|
| fiction | 22.124 | 1.780 | 3.519 | 3.762 |
| code | 102.903 | 2.100 | 5.910 | 6.103 |
| news | 32.285 | 2.489 | 3.419 | 5.288 |
| dialogue | 38.518 | 2.585 | 4.014 | 4.228 |
| universal | 22.124 | 1.780 | 3.519 | 3.762 |

