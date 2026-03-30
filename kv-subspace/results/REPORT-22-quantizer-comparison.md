# Experiment 22: Quantizer Comparison — SubRotQ vs PolarQuant

Model: Qwen/Qwen3-14B-AWQ
Baseline PPL: 1.1701

## Results

| k | bits | quantizer | PPL | rel-PPL |
|---|------|-----------|-----|---------|
| 64 | 4 | polarquant | 1.3396 | 1.1449 |
| 64 | 4 | subrotq | 1.2189 | 1.0417 |
| 64 | 8 | polarquant | 1.2067 | 1.0313 |
| 64 | 8 | subrotq | 1.2063 | 1.0310 |
| 96 | 4 | polarquant | 1.2433 | 1.0626 |
| 96 | 4 | subrotq | 1.1767 | 1.0057 |
| 96 | 8 | polarquant | 1.1669 | 0.9973 |
| 96 | 8 | subrotq | 1.1672 | 0.9975 |
| 112 | 4 | polarquant | 1.2351 | 1.0556 |
| 112 | 4 | subrotq | 1.1733 | 1.0028 |
| 112 | 8 | polarquant | 1.1713 | 1.0011 |
| 112 | 8 | subrotq | 1.1708 | 1.0006 |
| 128 | 4 | polarquant | 1.2654 | 1.0815 |
| 128 | 4 | subrotq | 1.1722 | 1.0018 |
| 128 | 8 | polarquant | 1.1705 | 1.0004 |
| 128 | 8 | subrotq | 1.1699 | 0.9998 |

## Key Questions

1. At matched (k, bits), does PolarQuant reduce PPL vs SubRotQ?
2. Is the quantizer gap constant across k values or does it grow at lower k?
3. Does the truncation-dominance finding hold: is the k=128→k=112 PPL
   gap larger than the SubRotQ→PolarQuant quantizer gap at any k?

## Quantizer Gap (PolarQuant rel_ppl - SubRotQ rel_ppl)

Negative = PolarQuant is better.

| k | bits | SubRotQ | PolarQuant | gap |
|---|------|---------|------------|-----|
| 64 | 4 | 1.0417 | 1.1449 | +0.1032 |
| 64 | 8 | 1.0310 | 1.0313 | +0.0003 |
| 96 | 4 | 1.0057 | 1.0626 | +0.0569 |
| 96 | 8 | 0.9975 | 0.9973 | -0.0002 |
| 112 | 4 | 1.0028 | 1.0556 | +0.0528 |
| 112 | 8 | 1.0006 | 1.0011 | +0.0005 |
| 128 | 4 | 1.0018 | 1.0815 | +0.0797 |
| 128 | 8 | 0.9998 | 1.0004 | +0.0006 |
