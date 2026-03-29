# Experiment 17: Cross-Domain Calibration Robustness

- Model: Qwen3-14B-AWQ
- Calibration tokens: 1024 per domain
- Eval context: 2048 tokens
- Domains: ['fiction', 'code', 'news', 'dialogue']

## PPL matrix — baseline

| calib↓ / eval→ | fiction | code | news | dialogue |
|---|---|---|---|---|
| fiction | 10.530 | 1.171 | 1.590 | 2.058 |
| code | 1.232 | 1.171 | 1.590 | 2.058 |
| news | 1.232 | 1.171 | 1.590 | 2.058 |
| dialogue | 1.232 | 1.171 | 1.590 | 2.058 |
| universal | 1.232 | 1.171 | 1.590 | 2.058 |

## PPL matrix — k128_4bit

| calib↓ / eval→ | fiction | code | news | dialogue |
|---|---|---|---|---|
| fiction | 10.963 | 1.191 | 1.621 | 2.084 |
| code | 1.235 | 1.206 | 1.609 | 2.077 |
| news | 1.276 | 1.199 | 1.612 | 2.079 |
| dialogue | 1.253 | 1.234 | 1.601 | 2.115 |
| universal | 1.265 | 1.191 | 1.621 | 2.084 |

## PPL matrix — k96_4bit

| calib↓ / eval→ | fiction | code | news | dialogue |
|---|---|---|---|---|
| fiction | 18.431 | 1.448 | 1.842 | 2.518 |
| code | 1.750 | 1.624 | 2.697 | 3.440 |
| news | 1.582 | 1.926 | 1.872 | 3.132 |
| dialogue | 1.642 | 1.926 | 2.156 | 2.563 |
| universal | 1.374 | 1.448 | 1.842 | 2.518 |

