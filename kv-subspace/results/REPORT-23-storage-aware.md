# Experiment 23: Storage-Aware Quantizer Comparison

Model: Qwen/Qwen3-14B-AWQ
Baseline PPL: 1.1701

## Storage Cost Summary

For d=128, n_bits=4:

| Method | Group Size | Effective bpe | True CR |
|--------|-----------|--------------|--------|
| SubRotQ | G=1    | 68.0 | 0.235x |
| SubRotQ | G=8     | 12.00  | 1.333x |
| SubRotQ | G=16    | 8.00  | 2.000x |
| SubRotQ | G=64    | 5.00  | 3.200x |
| SubRotQ | G=128   | 4.50  | 3.556x |
| PolarQuant | per-vec | 4.22 | 3.793x |
| FP16 baseline | — | 16.00 | 1.000x |

## Quality vs True Compression Ratio

| k | bits | method | group | eff CR | rel-PPL |
|---|------|--------|-------|--------|--------|
| 112 | 4 | polarquant | 1 | 4.303x | 1.0556 |
| 112 | 4 | subrotq | 1 | 0.269x | 1.0010 |
| 112 | 4 | subrotq | 8 | 1.524x | 0.9984 |
| 112 | 4 | subrotq | 16 | 2.286x | 0.9978 |
| 112 | 4 | subrotq | 64 | 3.657x | 1.0047 |
| 112 | 4 | subrotq | 128 | 4.064x | 1.0008 |
| 112 | 4 | subrotq | 512 | 4.433x | 1.0041 |
| 112 | 4 | subrotq | 2048 | 4.536x | 1.0029 |
| 112 | 8 | polarquant | 1 | 2.226x | 1.0011 |
| 112 | 8 | subrotq | 1 | 0.254x | 1.0010 |
| 112 | 8 | subrotq | 8 | 1.143x | 1.0010 |
| 112 | 8 | subrotq | 16 | 1.524x | 1.0013 |
| 112 | 8 | subrotq | 64 | 2.032x | 1.0011 |
| 112 | 8 | subrotq | 128 | 2.151x | 1.0013 |
| 112 | 8 | subrotq | 512 | 2.251x | 1.0008 |
| 112 | 8 | subrotq | 2048 | 2.277x | 1.0010 |
| 128 | 4 | polarquant | 1 | 3.793x | 1.0815 |
| 128 | 4 | subrotq | 1 | 0.235x | 0.9999 |
| 128 | 4 | subrotq | 8 | 1.333x | 1.0001 |
| 128 | 4 | subrotq | 16 | 2.000x | 0.9991 |
| 128 | 4 | subrotq | 64 | 3.200x | 1.0025 |
| 128 | 4 | subrotq | 128 | 3.556x | 0.9976 |
| 128 | 4 | subrotq | 512 | 3.879x | 1.0010 |
| 128 | 4 | subrotq | 2048 | 3.969x | 1.0027 |
| 128 | 8 | polarquant | 1 | 1.954x | 1.0004 |
| 128 | 8 | subrotq | 1 | 0.222x | 0.9999 |
| 128 | 8 | subrotq | 8 | 1.000x | 0.9998 |
| 128 | 8 | subrotq | 16 | 1.333x | 1.0001 |
| 128 | 8 | subrotq | 64 | 1.778x | 1.0000 |
| 128 | 8 | subrotq | 128 | 1.882x | 0.9998 |
| 128 | 8 | subrotq | 512 | 1.969x | 0.9998 |
| 128 | 8 | subrotq | 2048 | 1.992x | 0.9997 |

## Key Findings

1. **SubRotQ G=1** (autoregressive reality): stores scale/offset per token,
   resulting in ~0.24x 'compression ratio' (actually expansion).
2. **SubRotQ G≥64** approaches PolarQuant storage cost.
3. **PolarQuant** achieves true per-vector quantization with no overhead —
   the key practical advantage over SubRotQ for deployment.
4. At matched *effective* CR, which achieves better PPL quality?
