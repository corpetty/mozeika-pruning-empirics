# Task C1: Full-N Downstream Tasks

Model: Qwen/Qwen3-14B-AWQ | N=1000/task | Wall: 553.0 min

## Accuracy Table

| Config | arc_challenge | hellaswag | arc_easy | winogrande |
|--------|--------|--------|--------|--------|
| baseline | 0.6840 | 0.7100 | 0.8160 | 0.7400 |
| k128_4bit | 0.6720 | 0.6990 | 0.8120 | 0.7240 |
| k96_4bit | 0.5540 | 0.6010 | 0.6080 | 0.6550 |

## Relative Accuracy (vs baseline)

| Config | arc_challenge | hellaswag | arc_easy | winogrande |
|--------|--------|--------|--------|--------|
| k128_4bit | -0.0120 | -0.0110 | -0.0040 | -0.0160 |
| k96_4bit | -0.1300 | -0.1090 | -0.2080 | -0.0850 |

_Calibration: WikiText-2 train split (2048 tokens). No overlap with downstream task data._
