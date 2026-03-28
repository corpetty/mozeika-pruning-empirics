# Experiment 6: End-to-End Perplexity with Compressed KV Cache

## Setup

- Model: Qwen3-14B-AWQ
- Evaluation: 3 text passages (scientific, historical, philosophical)
- Sequence length: 512 tokens
- Compression applied to k_proj/v_proj outputs via forward hooks

## Compression Configs

| Config | K compression | V compression |
|--------|--------------|--------------|
| baseline | none | none |
| K_sub_k64_4bit | subspace k=64, 4-bit | none |
| KV_optimal | subspace k=64, 4-bit | full-dim 4-bit |
| KV_aggressive | subspace k=64, 2-bit | full-dim 2-bit |

## Perplexity Results

| Config | P0 | P1 | P2 | Mean PPL | Rel. PPL |
|--------|--------|--------|--------|----------|----------|
| baseline | 2.09 | 2.93 | 2.73 | 2.58 | 1.0000 |
| K_sub_k64_4bit | 4.82 | 10.62 | 7.88 | 7.78 | 2.9406 |
| KV_optimal | 4.93 | 11.45 | 8.38 | 8.25 | 3.1108 |
| KV_aggressive | 27.50 | 113.81 | 39.78 | 60.36 | 22.1865 |

## PPL Degradation vs Baseline

- **K_sub_k64_4bit**: +194.06% — EXCEEDS 5% threshold
- **KV_optimal**: +211.08% — EXCEEDS 5% threshold
- **KV_aggressive**: +2118.65% — EXCEEDS 5% threshold

## KL Proxy Correlation

Previous experiments measured KL divergence on reconstructed KV vectors as a proxy for compression quality. The expected rank order of degradation (from proxy KL, least to most aggressive):
1. K_sub_k64_4bit (K-only, least aggressive)
2. KV_optimal (K+V at 4-bit)
3. KV_aggressive (K+V at 2-bit, most aggressive)

Actual PPL degradation rank order (least to most):
1. K_sub_k64_4bit: +194.06%
2. KV_optimal: +211.08%
3. KV_aggressive: +2118.65%

The PPL rank order **matches** the KL proxy rank order — proxy is validated.

## Conclusion

- **K_sub_k64_4bit** exceeds the 5% PPL threshold — not recommended without further tuning.
- **KV_optimal** exceeds the 5% PPL threshold — not recommended without further tuning.
- **KV_aggressive** exceeds the 5% PPL threshold — not recommended without further tuning.
