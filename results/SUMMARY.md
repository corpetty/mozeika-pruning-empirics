# KV Cache Subspace Compression: Research Summary

## Abstract

We investigate compressing the key-value (KV) cache of large language models by projecting KV vectors into lower-dimensional PCA subspaces before quantizing with PolarQuant. Across 11 experiments on Qwen3 models (1.7B to 32B parameters), we find that **truncation error from dimension reduction dominates quantization noise** — the critical threshold is retaining at least 87.5% of dimensions (k/d_head >= 0.875). The best configuration for Qwen3-14B-AWQ is k=112/4-bit, which achieves 4.27x compression with only 14% perplexity increase, while k=128/4-bit (pure PolarQuant, no truncation) achieves 4.00x compression at just 5% PPL cost. Larger models tolerate more aggressive truncation: Qwen3-32B needs only k=96, while Qwen3-1.7B requires k=128.

## 1. What We Are Trying To Do

Modern LLMs store a key and value vector for every token at every layer during inference. For Qwen3-14B with a 4K-token context, this KV cache consumes ~168 MB in FP16 — and it scales linearly with sequence length. Long-context applications (RAG, multi-turn chat, document analysis) are often memory-bound by the KV cache, not by model weights.

The goal is to compress these KV vectors at write time (when a new token enters the cache) and decompress at read time (when computing attention). The compression must be fast enough to not bottleneck decoding and faithful enough to preserve the model's output quality.

Our approach combines two ideas: (1) **PCA subspace projection**, which projects each d_head=128 dimensional KV vector down to k < 128 dimensions using a pre-computed orthogonal basis, and (2) **PolarQuant**, a rotation-based quantization method that applies a learned rotation before scalar quantization to decorrelate dimensions. The projection captures the dominant variance directions; the quantization compresses the projected coordinates to low bit-width.

## 2. How It Works

Each KV head in each layer produces 128-dimensional key and value vectors. We fit a PCA basis offline on calibration text: for a given (layer, head) pair, we collect KV vectors, compute the principal components, and store the top-k eigenvectors as a projection matrix W of shape (128, k).

At inference time, for each new KV vector x (128-dim):
1. **Project**: z = W^T x (128 -> k dims, captures top-k variance directions)
2. **Rotate**: z' = R z (k x k rotation from PolarQuant, decorrelates for quantization)
3. **Quantize**: q = round(z' / scale) (uniform scalar quantization to n bits)
4. **Store**: q in the KV cache (k * n bits instead of 128 * 16 bits)

At attention time, the reverse path reconstructs an approximate x-hat = W R^(-1) (q * scale). The compression ratio is (128 * 16) / (k * n) — for k=112, n=4, this is 4096/448 = 4.27x.

The key insight is that KV vectors are low-rank: their effective dimensionality (the number of principal components needed to capture 90% of variance) averages only 30 for K vectors and 54 for V vectors across the 40 layers of Qwen3-14B. This means most of the "information" lives in far fewer than 128 dimensions — but as our experiments show, the remaining 10% of variance that lives outside the top-k subspace turns out to matter more than expected.

## 3. The Story in Order

**Experiment 1** tested the Universal Weight Subspace Hypothesis (UWSH) — whether the PCA subspaces of different layers, heads, and text domains overlap enough to share projection bases. The answer was nuanced: cross-domain overlap for K vectors was 0.70 (good news for offline calibration), but cross-head overlap was only 0.46 (each head needs its own basis). Cross-layer overlap was moderate at 0.56 for K and 0.74 for V, with early layers (L0-9) being the least aligned. This told us that per-(layer, head) bases are necessary, but a single calibration run across domains should work.

**Experiment 2** asked whether adapting the subspace dimension k per head — setting it to the effective rank at 90% variance (k = eff_rank_90, ranging from 7 to 52) — would outperform a fixed k=64. It did not. Fixed k=64 won on 100% of the 320 layer-head combinations, by a factor of 3.4x at the 2-bit budget and 6.2x at 4-bit. The reason: eff_rank_90 discards 10% of variance permanently, and that irrecoverable truncation loss outweighs any gain from higher per-dimension bit precision. This was our first hint that truncation error is the binding constraint.

**Experiment 3** revealed an asymmetry between K and V vectors. V vectors have nearly 2x the effective rank of K (mean 54 vs 30), making them harder to compress via subspace projection. At k=64, subspace projection captures >99% of K variance but only ~90% of V variance. The practical implication: K benefits strongly from subspace compression, V benefits less. A hybrid strategy — subspace for K, full-dim quantization for V — emerged as natural.

**Experiment 4** validated offline calibration transfer. A PCA basis fitted on Project Gutenberg fiction text and applied to Wikipedia-style factual text incurred a 2.06x KL divergence penalty versus an oracle (same-domain) basis at 2-bit — but still beat full-dimensional quantization on 99.7% of heads. At 4-bit, the transfer penalty grew to 3.8x and full-dim won on 91% of heads. The takeaway: offline calibration is viable at low bit rates (where subspace compression is most valuable), but at higher bit rates, just use full-dim quantization.

**Experiment 5** measured hardware overhead on an RTX 3090. The full subspace-PolarQuant pipeline (project + rotate + quantize + inverse) costs 1.76x the latency of plain quantization — about 325 microseconds per head versus 185 microseconds at T=512. This overhead is dominated by kernel launch latency, not compute, and could be reduced to ~1.2x with a fused CUDA kernel. For long-context inference where memory savings dominate, this overhead is acceptable; for real-time single-token generation, it adds roughly 5 ms across 40 layers.

**Experiment 6** was the first end-to-end perplexity test — and the results were sobering. With k=64/4-bit on K and full-dim 4-bit on V (the "optimal" config from distortion analysis), mean PPL jumped from 2.58 to 8.25, a 3.19x degradation. Even K-only compression at k=64 produced 2.94x PPL. The aggressive 2-bit config exploded to 22x PPL. Clearly, k=64 was too aggressive despite looking reasonable on per-layer distortion metrics.

**Experiment 7** explained why. Attention fidelity analysis showed that compressed KV vectors at k=64 preserved the correct top-1 attention target on only 37% of tokens overall. Early layers maintained 63% fidelity, but mid layers dropped to 29% and late layers to 26%. The compression errors accumulated layer-by-layer, with each layer's distorted KV output feeding into the next layer's attention computation — a cascade effect.

**Experiment 8** tested the natural remedy: protect early layers from compression and only compress later ones. Protecting layers 0-19 (half the model) reduced PPL from 8.25 to 4.53 — a 66% recovery of the gap. But even this best mixed policy still produced 75% PPL degradation, and the compression ratio dropped to only 1.68x. The conclusion was clear: the problem was not just cascading errors — k=64 introduced too much per-layer error.

**Experiment 9** was the breakthrough. A systematic sweep across k={64, 96, 112, 128} and bit depths {4, 6, 8, 16} revealed that **truncation error dominates quantization noise**. The smoking gun: k=64/16-bit (pure truncation, near-lossless quantization) gave 2.48x PPL, while k=128/4-bit (no truncation, aggressive quantization) gave only 1.05x PPL. Retaining all 128 dimensions matters far more than having high bit precision. The Pareto frontier ran along the 4-bit column: k=112/4-bit (1.14x PPL, 4.27x CR) and k=128/4-bit (1.05x PPL, 4.00x CR) emerged as the practical sweet spots.

**Experiment 10** confirmed that at k=112, the cascade problem from Experiment 8 largely disappears. Uniform k=112/4-bit compression across all 40 layers achieved 1.14x PPL — within the 20% threshold. Mixed policies (protecting early layers) offered marginal PPL improvements (1.11-1.15x) but at severe compression cost (dropping from 4.27x to 1.62-2.35x). The hybrid config of k=128 for early layers and k=112 for late layers achieved 1.10x PPL at 4.13x CR — a clean middle ground. The key insight: when per-layer error is small enough, 40-layer accumulation stays within budget, and mixed policies become unnecessary complexity.

**Experiment 11** tested generalization across model sizes. The k/d_head >= 0.875 rule that worked for the 14B model did not universally hold. Qwen3-1.7B needed k/d_head = 1.0 (full dimensions) to stay within 20% PPL — smaller models are more sensitive to truncation. Qwen3-32B-AWQ was the most tolerant, achieving 1.09x PPL at k/d_head = 0.75 (k=96). The pattern is monotonic: larger models tolerate more aggressive subspace compression. At k/d_head = 0.875, rel PPL was 1.30x for 1.7B, 1.14x for 14B, and 1.06x for 32B.

## 4. The Key Results Table

| Config | k | Bits | Rel PPL | Compression | Best For |
|--------|---|------|---------|-------------|----------|
| k128/4-bit uniform | 128 | 4 | 1.05x | 4.00x | Minimum PPL impact |
| hybrid k128-early/k112-late | 128/112 | 4 | 1.10x | 4.13x | Balanced |
| k112/4-bit uniform | 112 | 4 | 1.14x | 4.27x | Maximum compression (<20% PPL) |
| k96/4-bit uniform | 96 | 4 | 1.26x | 4.57x | Aggressive (>20% PPL budget) |
| k64/4-bit uniform | 64 | 4 | 3.19x | 5.33x | Not recommended |

All configs use K subspace + V full-dim PolarQuant at 4 bits. Model: Qwen3-14B-AWQ, d_head=128.

## 5. What We Learned About Subspace Structure

The PCA subspaces of KV vectors reveal a rich geometric structure that directly determines compression viability.

**Effective rank asymmetry.** K vectors have a mean effective rank (90% variance) of 29.6 across 320 (layer, head) pairs, ranging from 7 in early layers to 52 in late layers. V vectors are nearly twice as distributed: mean effective rank 54.2, ranging from 13 to 79. This explains why K compresses well via subspace projection (k=64 captures >99% of K variance) while V does not (k=64 captures only ~90% of V variance). The practical consequence is asymmetric treatment: project K into a subspace, quantize V at full dimensionality.

**Cross-head independence.** With a mean overlap of only 0.46 for K and 0.55 for V, different attention heads within the same layer learn substantially different subspaces. This rules out sharing a single projection basis across heads — each of the 8 KV heads per layer requires its own PCA matrix. The total basis storage for Qwen3-14B is 40 layers x 8 heads x 128 x k floats, which at k=112 is about 45 MB in FP32 — a fixed overhead that is small relative to the KV cache savings at long context lengths.

**Layer-depth gradient.** Subspace overlap increases with depth. Early layers (L0-9) show cross-layer overlap of 0.38 for K and 0.68 for V; late layers (L30-39) reach 0.65 for K and 0.76 for V. This correlates with the effective rank distribution: early layers have more variable, input-dependent representations, while deep layers converge to more stable geometric patterns. Compression difficulty also follows this gradient — late layers have the highest effective ranks and the largest attention KL under compression.

**The UWSH partially holds.** The Universal Weight Subspace Hypothesis predicts significant overlap between subspaces of different components. For KV caches, this holds moderately across layers (0.56-0.74) and across domains (0.64-0.70), but breaks down across heads (0.46-0.55). The practical upshot: calibration transfer across text domains works, but structural sharing across heads does not.

## 6. The Calibration Story

A realistic deployment requires computing PCA bases offline on calibration text and applying them to arbitrary user inputs at inference time. Experiment 4 tested this by fitting bases on fiction and evaluating on factual text.

At the 2-bit budget (where subspace compression delivers its largest advantage over full-dim), cross-domain transfer incurs a 2.06x KL penalty versus an oracle basis — but still beats full-dimensional quantization on 99.7% of heads. The transfer penalty is roughly constant across layers (~2x) and only slightly higher in early layers (2.14x) versus late layers (1.96x), consistent with the cross-domain overlap of 0.70 measured in Experiment 1.

At higher bit rates (4-bit), the calculus reverses: full-dim quantization becomes competitive, and the transfer penalty grows to 3.8x. Since our recommended operating point is 4-bit (not 2-bit), and we recommend full-dim quantization for V vectors regardless, the calibration transfer question is most relevant for K vectors at aggressive compression levels.

The practical recommendation: a single offline calibration run on a few thousand tokens of diverse text produces PCA bases that generalize well across domains. No per-domain or per-user recalibration is needed.

## 7. Root Cause: Truncation vs Quantization

The critical insight from Experiment 9 is the relative magnitude of two error sources:

**Truncation error** arises from projecting 128-dim vectors down to k < 128 dimensions, permanently discarding the bottom (128-k) principal components. At k=64, this discards ~1% of K variance and ~10% of V variance — but that "small" percentage translates to large PPL impact because the discarded components, while individually small, collectively encode fine-grained distinctions that attention relies on.

**Quantization error** arises from representing continuous values with discrete bit levels. At 4-bit (16 levels), the quantization noise per dimension is small relative to the signal. PolarQuant's rotation step further reduces quantization error by decorrelating dimensions before quantization.

The smoking-gun comparison: k=64/16-bit (pure truncation, negligible quantization noise) produces 2.48x PPL, while k=128/4-bit (zero truncation, aggressive quantization) produces only 1.05x PPL. Truncation error is **24x worse** in PPL impact than quantization error. Furthermore, for k=64, increasing bit depth from 4 to 16 barely helps (PPL drops from 3.19x to 2.48x) because the truncation floor dominates.

This explains the failure of k=64 in all end-to-end experiments (Experiments 6-8) and why the solution is not more bits but more dimensions: k=112/4-bit vastly outperforms k=64/8-bit despite using fewer total bits per vector (448 vs 512).

## 8. Cross-Model Generalization

Experiment 11 tested whether the k/d_head >= 0.875 threshold found on Qwen3-14B generalizes to other model sizes, all with d_head=128 and 4-bit PolarQuant.

| Model | Params | k/d_head for <20% PPL | Best rel PPL at k=112 | Best rel PPL at k=128 |
|-------|--------|----------------------|----------------------|----------------------|
| Qwen3-1.7B | 1.7B | 1.0 (k=128) | 1.30x | 1.13x |
| Qwen3-14B-AWQ | 14B | 0.875 (k=112) | 1.14x | 1.05x |
| Qwen3-32B-AWQ | 32B | 0.75 (k=96) | 1.06x | 1.04x |

The pattern is clear: larger models tolerate more aggressive subspace truncation. Qwen3-32B achieves only 1.09x PPL at k=96 (0.75 of d_head), where the 14B model would show 1.26x and the 1.7B model would be at 1.80x. This likely reflects the over-parameterization of larger models — they encode more redundant information per head, making the KV subspaces genuinely lower-rank in a functional sense.

The practical implication: the subspace dimension k should be tuned per model size. A safe rule of thumb is k/d_head >= 0.875 for models above ~10B parameters, and k/d_head = 1.0 (full-dim PolarQuant only) for models below ~5B parameters.

## 9. What This Means for Practitioners

**KV cache compression at 4x is practical for large models.** Qwen3-14B-AWQ with k=112/4-bit achieves 4.27x compression (960 bits per KV pair vs 4096 bits at FP16) with only 14% PPL increase. For Qwen3-32B, 4.57x compression is achievable at similar quality.

**Don't truncate too aggressively.** The biggest mistake is setting k too low. k=64 (half the dimensions) looks appealing on paper but causes 3x PPL blowup. The truncation floor means no amount of quantization precision can compensate. Stay above k/d_head = 0.875 for models >10B.

**Treat K and V differently.** K vectors are genuinely low-rank (eff_rank ~30) and benefit strongly from subspace projection. V vectors are higher-rank (eff_rank ~54) and are better served by full-dim quantization at the same bit rate. The recommended pipeline is: K via subspace PolarQuant (project to k dims, rotate, quantize), V via full-dim PolarQuant (rotate all 128 dims, quantize).

**Offline calibration works.** A single calibration pass on a few thousand tokens of generic text produces PCA bases that transfer across domains with modest (~2x KL) penalty. No online adaptation is needed.

**Hardware overhead is non-trivial but manageable.** The subspace projection adds ~1.7x latency per head (325 vs 185 microseconds). Across 40 layers, this is ~5 ms per token — significant for real-time chat but amortized in batch/long-context settings. A fused CUDA kernel could reduce this to ~1.2x.

**Mixed layerwise policies are unnecessary at k >= 112.** When per-layer error is small (as with k=112), the cascade effect is negligible and uniform compression across all layers is optimal. Protecting early layers helps marginally at k=64 but buys almost nothing at k=112 while severely reducing compression ratio.

## 10. Recommended Configurations

### For Qwen3-14B-AWQ (d_head=128)

| Priority | Config | k | Bits | K Strategy | V Strategy | Rel PPL | CR |
|----------|--------|---|------|------------|------------|---------|-----|
| Safety | k128/4-bit | 128 | 4 | Full-dim PolarQuant | Full-dim PolarQuant | 1.05x | 4.00x |
| Balanced | Hybrid | 128/112 | 4 | k=128 L0-19, k=112 L20-39 | Full-dim PolarQuant | 1.10x | 4.13x |
| Compression | k112/4-bit | 112 | 4 | Subspace PolarQuant | Full-dim PolarQuant | 1.14x | 4.27x |

### For Other Model Sizes (4-bit, uniform policy)

| Model | Recommended k | k/d_head | Expected Rel PPL | CR |
|-------|--------------|----------|------------------|-----|
| Qwen3-1.7B | 128 | 1.0 | ~1.13x | 4.00x |
| Qwen3-14B-AWQ | 112 | 0.875 | ~1.14x | 4.27x |
| Qwen3-32B-AWQ | 96 | 0.75 | ~1.09x | 4.57x |

### Hardware Requirements

- **Basis storage**: 40 layers x 8 heads x 128 x k x 4 bytes = ~45 MB at k=112
- **Calibration**: One forward pass on ~2K tokens of generic text
- **Latency overhead**: ~1.7x per-head (reducible to ~1.2x with fused kernels)
- **Memory savings at 4K context**: 168 MB -> ~40 MB (k=112/4-bit)

## 11. Open Questions and Next Steps

**Fused CUDA kernels.** The current 1.7x latency overhead is dominated by kernel launch costs, not compute. A single fused kernel for project-rotate-quantize could reduce this to 1.2-1.3x, making the overhead negligible.

**Different d_head values.** All experiments used d_head=128. Models with d_head=64 (e.g., some Llama variants) have a different rank structure and may need different k/d_head thresholds. Models with d_head=256 might tolerate more aggressive truncation.

**Learned projection bases.** PCA is an unsupervised projection optimizing reconstruction MSE. A task-aware projection (e.g., trained to minimize attention KL directly) could potentially achieve the same PPL at lower k, but adds training complexity.

**Token-level adaptive compression.** All experiments compress every token uniformly. High-attention tokens (anchors, special tokens) may benefit from lossless or higher-fidelity storage, while low-attention tokens could tolerate more aggressive compression.

**Scaling to 100K+ contexts.** The KV cache savings become most impactful at very long context lengths, where memory is the primary bottleneck. Testing at 100K+ tokens would validate the practical benefit and may reveal new phenomena (e.g., whether the PCA basis drifts over very long sequences).

**Integration with other compression methods.** Subspace PolarQuant could be combined with token eviction, sliding window attention, or grouped-query attention for compound memory savings. The interaction between these techniques is unexplored.

**Non-uniform bit allocation.** Experiments used the same bit depth for all layers and heads. Allocating more bits to high-effective-rank late layers and fewer bits to low-rank early layers could improve the overall PPL-compression tradeoff.
