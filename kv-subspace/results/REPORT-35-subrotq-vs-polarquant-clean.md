# Task B1: SubRotQ vs PolarQuant (clean WikiText-2)

Model: Qwen/Qwen3-14B-AWQ | Baseline PPL: 6.5676 | Wall: 1.6 min

| Method | k | bits | PPL | rel PPL |
|--------|---|------|-----|------|
| baseline | 128 | 16 | 6.5676 | 1.0 |
| plain_4bit | 128 | 4 | 6.4974 | 0.9893 |
| subrotq | 112 | 4 | 8.0759 | 1.2297 |
| polarquant | 112 | 4 | 12.5782 | 1.9152 |
| subrotq | 128 | 4 | 6.4421 | 0.9809 |
| polarquant | 128 | 4 | 11.817 | 1.7993 |

## Key comparisons

SubRotQ k=128: 0.9809x vs PolarQuant k=128: 1.7993x  (diff: -0.8184x)

**SubRotQ wins by 0.8184x rel PPL at k=128.**

SubRotQ k=128 vs plain 4-bit: -0.0084x rel PPL gap.
At full rank, SubRotQ and plain quant are indistinguishable.

## Prior exp22 reference (contaminated pipeline)
exp22 reported SubRotQ k=112/4-bit: 1.0028x vs PolarQuant: 1.0556x on War & Peace.
This experiment provides the clean-benchmark equivalent.
