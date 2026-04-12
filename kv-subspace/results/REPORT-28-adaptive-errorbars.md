# Experiment 28: Adaptive Scheduling Error Bars

N_SEEDS=5, budgets=[96, 104, 112], WikiText-2 calib/eval


## Summary (mean ± std across seeds)

| Budget k | Policy | Mean k | Rel PPL (mean) | Rel PPL (std) |
|----------|--------|--------|----------------|---------------|
| 96 | uniform | 96.0 | 1.4989 | ±0.0550 |
| 96 | rank_prop | 96.0 | 2.0286 | ±0.1009 |
| 104 | uniform | 96.0 | 1.4989 | ±0.0550 |
| 104 | rank_prop | 96.0 | 2.0286 | ±0.1009 |
| 112 | uniform | 112.0 | 1.2290 | ±0.0349 |
| 112 | rank_prop | 110.0 | 1.8723 | ±0.1057 |

## Key Question
Does rank-proportional scheduling consistently beat uniform k across calibration seeds,
or was exp18's result a lucky draw?

## Conclusion
Rank-proportional beats uniform in 0/3 budget points (mean across 5 seeds).