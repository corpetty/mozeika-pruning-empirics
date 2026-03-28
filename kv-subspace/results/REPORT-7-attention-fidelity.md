# Experiment 7: Attention Score Fidelity Under KV Compression

## Setup

- Model: Qwen3-14B-AWQ
- 40 layers, 40 query heads, 8 KV heads, d_head=128
- Compression config: K subspace k=64 4-bit, V full-dim 4-bit (KV_optimal)
- Attention computed from pre-RoPE Q,K projections (isometry preserves relative comparison)
- Sequence length: 512 tokens

## Top-1 Match Rate by Layer Range

| Layer range | Token range | Top-1 match | Top-5 Jaccard | Attn KL |
|-------------|-------------|-------------|---------------|---------|
| early (L0-9) | early | 0.6692 | 0.6681 | 0.000836 |
| early (L0-9) | late | 0.5908 | 0.5552 | 0.000892 |
| mid (L10-29) | early | 0.3380 | 0.3880 | 1.263898 |
| mid (L10-29) | late | 0.2486 | 0.2593 | 1.772670 |
| late (L30-39) | early | 0.2972 | 0.3404 | 11.933515 |
| late (L30-39) | late | 0.2172 | 0.2363 | 13.825333 |

## Per-Layer Aggregation (all tokens)

| Layer | Top-1 match | Top-5 Jaccard | Attn KL |
|-------|-------------|---------------|---------|
|  0 | 0.8152 | 0.7062 | 0.000000 |
|  1 | 0.6674 | 0.6279 | 0.000003 |
|  2 | 0.6768 | 0.6452 | 0.000009 |
|  3 | 0.6324 | 0.6234 | 0.000027 |
|  4 | 0.5998 | 0.5798 | 0.000076 |
|  5 | 0.5890 | 0.5599 | 0.000072 |
|  6 | 0.5945 | 0.6085 | 0.000169 |
|  7 | 0.6539 | 0.6468 | 0.001858 |
|  8 | 0.5851 | 0.5707 | 0.004720 |
|  9 | 0.4858 | 0.5481 | 0.001706 |
| 10 | 0.4719 | 0.4690 | 0.007525 |
| 11 | 0.4657 | 0.4673 | 0.014161 |
| 12 | 0.3727 | 0.4012 | 0.012665 |
| 13 | 0.3472 | 0.3816 | 0.057974 |
| 14 | 0.4295 | 0.4312 | 0.035191 |
| 15 | 0.3303 | 0.3601 | 0.029210 |
| 16 | 0.3186 | 0.3307 | 0.068048 |
| 17 | 0.2989 | 0.3582 | 0.075693 |
| 18 | 0.2218 | 0.2652 | 0.146076 |
| 19 | 0.2304 | 0.2850 | 0.666153 |
| 20 | 0.1937 | 0.2533 | 0.791480 |
| 21 | 0.2197 | 0.2705 | 0.769294 |
| 22 | 0.2020 | 0.2579 | 2.006386 |
| 23 | 0.2209 | 0.2555 | 1.431212 |
| 24 | 0.1482 | 0.2009 | 4.939697 |
| 25 | 0.1583 | 0.2066 | 4.516565 |
| 26 | 0.3367 | 0.3094 | 2.088170 |
| 27 | 0.3258 | 0.3097 | 5.626050 |
| 28 | 0.2699 | 0.3196 | 3.356005 |
| 29 | 0.3044 | 0.3396 | 3.728132 |
| 30 | 0.2846 | 0.3311 | 6.919839 |
| 31 | 0.2089 | 0.2916 | 8.091719 |
| 32 | 0.3238 | 0.3358 | 9.342944 |
| 33 | 0.2619 | 0.2914 | 11.794620 |
| 34 | 0.2182 | 0.2482 | 15.646902 |
| 35 | 0.2221 | 0.2556 | 16.127511 |
| 36 | 0.2316 | 0.2596 | 15.868419 |
| 37 | 0.2393 | 0.2861 | 15.939847 |
| 38 | 0.3435 | 0.3428 | 14.091386 |
| 39 | 0.2379 | 0.2408 | 14.971053 |

## Do Late Layers (30-39) Show Worst Fidelity?

- **early (L0-9)**: mean attn KL = 0.000864, top-1 match = 0.6300
- **mid (L10-29)**: mean attn KL = 1.518284, top-1 match = 0.2933
- **late (L30-39)**: mean attn KL = 12.879424, top-1 match = 0.2572

Yes — late layers show the highest attention KL, consistent with KV compression experiments showing late layers are hardest to compress.

## Practical Implication

- Overall top-1 match rate: **0.3685** (36.8% of tokens attend to the same top token)
- Overall top-5 Jaccard: **0.3868**
- Overall attention KL: **3.979214**

Compression significantly changes attention patterns. Consider less aggressive compression for layers with low fidelity.
