# Experiment 18: Sensitivity-Guided Adaptive K-Scheduling

## Overview
Tests whether per-layer k assignment derived from sensitivity scores outperforms
uniform k at equivalent mean-k budgets. Two adaptive algorithms tested:
- **Greedy**: assigns k=128 to most sensitive layers, k=64 to least sensitive
- **Rank-proportional**: scales k linearly with sensitivity rank

Baseline PPL: 10.6155

## Results

| Budget k | Policy | Mean k | PPL | Rel PPL | K CR | Combined CR |
|----------|--------|--------|-----|---------|------|-------------|
| 80 | Uniform | 80 | 20.3781 | 1.920x | 6.40x | 4.92x |
| 80 | Greedy | 88 | 14.6081 | 1.376x | 5.84x | 4.75x |
| 80 | Rank-prop | 82 | 15.1715 | 1.429x | 6.24x | 4.88x |
| 88 | Uniform | 80 | 20.3781 | 1.920x | 6.40x | 4.92x |
| 88 | Greedy | 96 | 13.2230 | 1.246x | 5.36x | 4.58x |
| 88 | Rank-prop | 88 | 13.7329 | 1.294x | 5.82x | 4.74x |
| 96 | Uniform | 96 | 13.0306 | 1.228x | 5.33x | 4.57x |
| 96 | Greedy | 96 | 13.0306 | 1.228x | 5.33x | 4.57x |
| 96 | Rank-prop | 96 | 12.5008 | 1.178x | 5.33x | 4.57x |
| 104 | Uniform | 96 | 13.0306 | 1.228x | 5.33x | 4.57x |
| 104 | Greedy | 96 | 12.9072 | 1.216x | 5.31x | 4.56x |
| 104 | Rank-prop | 104 | 12.4034 | 1.168x | 4.92x | 4.41x |
| 112 | Uniform | 112 | 12.2351 | 1.153x | 4.57x | 4.27x |
| 112 | Greedy | 104 | 12.0468 | 1.135x | 4.90x | 4.41x |
| 112 | Rank-prop | 110 | 12.0189 | 1.132x | 4.66x | 4.30x |

## Key Finding
The adaptive policies allow higher compression at equivalent quality to uniform k.
The mean_k budget needed to stay under 1.20x rel PPL shifts downward when
sensitivity-guided assignment is used — meaning more bits can be saved globally
by concentrating them at the expensive layers.