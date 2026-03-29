# Experiment 20: V-specific k Threshold Scan

**Model:** Qwen/Qwen3-14B-AWQ  
**K fixed at:** k=112/4-bit  
**Viability threshold:** gap_vs_K_only < 0.05x rel PPL  
**Baseline PPL (ctx=4096):** 6.3711  
**K-only reference PPL:** 7.1937 (rel=1.1291)

## Motivation

Exp 13 measured PCA basis overlap for K and V across context lengths and found:
- K: overlap = 0.825 (stable, low-dimensional structure)
- V: overlap = 0.702 (~30% of variance lives beyond the first 90 dimensions)

Exp 19 showed that online basis updating cannot close this gap — the failure is
structural, not a drift problem. This experiment determines the minimum k_V at
which V subspace compression becomes viable, and whether more quantization bits
can substitute for more dimensions.

## V k Scan Results (ctx=4096)

K fixed at k=112/4-bit throughout. Viability = gap < 0.05x above K-only reference.

| k_V | PPL | Rel PPL | Gap vs K-only | CR | Viable |
|-----|-----|---------|---------------|-----|--------|
| 64 | 35.76 | 5.61x | +4.48 | 5.82x | ✗ |
| 80 | 17.91 | 2.81x | +1.68 | 5.33x | ✗ |
| 96 | 10.75 | 1.69x | +0.56 | 4.92x | ✗ |
| 104 | 10.11 | 1.59x | +0.46 | 4.74x | ✗ |
| 108 | 10.06 | 1.58x | +0.45 | 4.65x | ✗ |
| 112 | 9.54 | 1.50x | +0.37 | 4.57x | ✗ |
| 116 | 8.84 | 1.39x | +0.26 | 4.49x | ✗ |
| 120 | 8.73 | 1.37x | +0.24 | 4.41x | ✗ |
| 124 | 8.68 | 1.36x | +0.23 | 4.34x | ✗ |
| **128** | **7.40** | **1.16x** | **+0.03** | **4.27x** | **✓** |

The curve does not show a sharp knee — degradation is roughly linear from k=96
to k=128. There is no "cheap" operating point: V requires the full subspace.

## Bits vs Dimensions Tradeoff

Can more bits compensate for fewer dimensions?

| Config | PPL | Rel PPL | Gap vs K-only | CR |
|--------|-----|---------|---------------|----|
| k_V=128 / 4-bit | 7.40 | 1.16x | +0.03 | 4.27x |
| k_V=112 / 4-bit | 9.54 | 1.50x | +0.37 | 4.57x |
| k_V=112 / 8-bit | 8.90 | 1.40x | +0.27 | 3.05x |

8-bit quantization at k=112 costs more memory than 4-bit at k=128 (3.05x vs 4.27x
compression) and still misses the viability threshold by 0.27. **More bits do not
substitute for more dimensions.**

## Long-Context Robustness (ctx=8192)

Only k_V=128 passed the short-context threshold, so it was tested at 8192 tokens:

| Config | Baseline PPL | K-only PPL | K+V PPL | Gap | Viable |
|--------|-------------|------------|---------|-----|--------|
| k_V=128 / 4-bit | 7.81 | 8.73 (1.12x) | 9.17 (1.17x) | +0.06 | ✗ |

Even full-rank V compression fails at 8K context. The gap widens from 0.03 to 0.06
as context grows, crossing the threshold. This is consistent with Exp 13's finding
that V basis overlap degrades with context length (0.702 at 4K, likely lower at 8K).

## Conclusion

**V subspace compression is not viable for Qwen3-14B-AWQ at any subspace dimension.**

The findings:

1. **No viable k_V exists below d_head=128.** PPL degrades roughly linearly from
   k=96 onward with no operating point inside the threshold.

2. **k_V=128 (full rank) passes at 4K ctx but fails at 8K.** The viable window is
   too narrow for production use.

3. **More bits don't help.** 8-bit at k=112 underperforms 4-bit at k=128 on both
   quality and compression ratio.

4. **The failure is architectural, not algorithmic.** Exp 19 ruled out drift (online
   updating does nothing). Exp 20 rules out insufficient dimensions. The 30% of V
   variance in the tail dimensions is load-bearing signal, not noise.

**Production recommendation:** Compress K only (k=112/4-bit, 4.27x CR on K cache,
1.14x PPL). Leave V at full precision. Combined KV savings are modest (~2x) but
quality is preserved.

## Future Work

The root cause of V's resistance to subspace compression compared to K remains
unexplained. Qwen3 applies QK-norm (RMSNorm after k_proj and q_proj) but not
V-norm — this may be the mechanism: K is normalized into a low-dimensional manifold
while V retains its full variance structure. Testing on architectures without QK-norm
(Llama, Mistral, Phi) would isolate this effect. Exp 17 showed Mistral tolerates
k=64 (5.33x CR, 1.12x PPL) which is promising, but V specifically was not ablated
there. A targeted cross-architecture V threshold scan across Llama-3/Mistral/Phi-3
would test whether QK-norm is the confounding variable and whether V compression is
viable in those families.
