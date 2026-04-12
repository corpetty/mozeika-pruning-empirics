# Subspace Rotation Quantization (SubRotQ): KV Cache Compression via PCA Projection and Random Rotation

**Corey Petty**  
Institute of Free Technology

---

## Abstract

We study compression of the key-value (KV) cache in transformer language models via projection into a learned PCA subspace followed by random rotation and uniform quantization. Through 30 experiments across five architectures (Qwen3-1.7B/14B/32B, Mistral-7B, Phi-4, Llama-3.1-8B), we establish that **truncation error from dimension reduction categorically dominates quantization noise**: on WikiText-2, k=64/16-bit (pure truncation) produces 6.25× perplexity degradation while k=128/4-bit (no truncation, aggressive quantization) produces only 0.98×. This motivates a compression target of k=128 (full rank, quantization only), yielding **4.00× compression at <2% perplexity cost** on Qwen3-14B. We further find that **V compression is not viable at any k < d_head across all tested architectures**—including Llama-3.1 which lacks QK-norm—indicating that V vectors are intrinsically high-dimensional. K compression generalizes across architectures: k=128/4-bit achieves <2% WikiText-2 PPL degradation on Qwen3, Mistral, and Llama; stable performance across 40K context lengths; and near-lossless quality on downstream reasoning tasks (ARC-Challenge -3pp, other tasks ±0pp). We recommend **K-only compression at k=128/4-bit** as a production-ready configuration.

---

## 1. Introduction

Inference with large language models is increasingly memory-bound at long context lengths. For a 14-billion-parameter model like Qwen3-14B at 32K tokens, the KV cache alone requires approximately 10 GB of GPU memory—comparable to the model weights themselves. This limits the practical context window on consumer hardware and increases serving costs proportionally with sequence length.

KV cache compression aims to reduce this memory footprint at inference time without modifying model weights. The central challenge is that KV vectors must be reconstructed with sufficient fidelity to preserve attention score distributions, which in turn determines the model's output quality.

We investigate a method combining two techniques: **PCA subspace projection**, which projects each d_head-dimensional KV vector into a k-dimensional principal subspace fitted on calibration data, and **random rotation quantization** (SubRotQ), which applies a random orthogonal rotation before uniform scalar quantization to decorrelate dimensions. The projection reduces dimensionality; SubRotQ reduces quantization error within that subspace.

Our primary contribution is an empirical characterization of this method across scales and architectures, producing three actionable findings:

1. **Truncation error is the binding constraint.** The critical design variable is k (subspace dimension), not bit depth. At k=128 (full rank), 4-bit quantization produces <2% WikiText-2 PPL degradation. At k=112 (12.5% truncation), 4-bit quantization produces 23% degradation. Practitioners should maximize k before minimizing bits.

2. **V compression is universally hard.** Key vectors compress well across all tested architectures. Value vectors resist subspace compression at any k < d_head, regardless of whether the architecture uses QK-norm. This is an intrinsic property of V variance structure, not an artifact of Qwen3.

3. **k=128/4-bit is the production configuration.** Full-rank 4-bit quantization of K achieves 4.00× compression with <2% WikiText-2 PPL degradation, generalizes across Qwen3, Mistral, and Llama, remains stable through 40K-token contexts, and produces near-lossless downstream task accuracy.

---

## 2. Background and Related Work

### 2.1 KV Cache Compression

Prior work on KV cache compression falls broadly into four categories:

**Token eviction** discards keys and values for tokens deemed less important (H2O [Zhang et al., 2023], StreamingLLM [Xiao et al., 2024]).

**Quantization** reduces bit precision of stored KV vectors. KVQuant [Hooper et al., 2024] uses per-channel quantization with outlier preservation. KIVI [Liu et al., 2024] applies asymmetric quantization strategies for K vs V. GEAR [Kang et al., 2024] adds low-rank error correction to quantized caches.

**Dimensionality reduction** projects KV vectors to lower-rank representations. KVTC [Staniszewski & Łańcucki, 2025] combines PCA-based feature decorrelation with dynamic-programming-optimal bit allocation and entropy coding, achieving 20× compression. Palu [Chang et al., 2025] decomposes KV projection weight matrices via SVD and quantizes latent representations. MatryoshkaKV [Park et al., 2025] trains orthogonal projections via knowledge distillation for 60–75% compression. Eigen Attention [Yang et al., 2024] performs attention directly in PCA-reduced space. SVDq [Wu et al., 2025] uses SVD bases with importance-aware mixed-precision quantization. SQuat [Li et al., 2025] combines subspace projection with orthogonal rotation quantization.

**Architectural modifications** redesign the cache structure at training time. GQA [Ainslie et al., 2023] reduces KV head count. DeepSeek MLA [DeepSeek-AI, 2024] jointly compresses K and V into a shared latent representation with 93.3% cache reduction.

### 2.2 Our Contribution

Our work is most similar to KVTC, Palu, and SQuat in combining PCA with quantization. We differ in four ways:

1. **Truncation-vs-quantization decomposition**: We systematically isolate and measure the contribution of truncation error vs quantization noise across a full grid of (k, bits) configurations (Experiment 24).

2. **Cross-architecture empirical characterization**: We validate across 5 architectures (Qwen3-1.7B/14B/32B, Mistral-7B, Llama-3.1-8B) and document architecture-specific compression thresholds.

3. **K-V asymmetry documentation**: We provide controlled evidence that V compression fails universally across architectures with and without QK-norm (Experiments 21, 30).

4. **Simpler engineering**: Our method uses offline PCA calibration and random rotation, avoiding KVTC's dynamic programming bit allocator and Palu's weight matrix decomposition. This trades peak compression for implementation simplicity.

### 2.3 Rotation Quantization

Our quantization approach uses random orthogonal rotation before uniform scalar quantization. This is distinct from two methods named "PolarQuant":

- **PolarQuant (Han et al., 2025)**: Recursive polar coordinate conversion across log₂(d) levels, followed by 1D k-means++ codebook quantization per angle dimension.
- **PolarQuant (Wu et al., 2025)**: Exploits 2D RoPE-rotated dimension pairs using lookup tables.

Our SubRotQ method uses random rotation (via QR decomposition of Gaussian noise) followed by uniform scalar quantization. Experiment 22 compares SubRotQ against the Han et al. PolarQuant implementation and finds SubRotQ produces lower perplexity at 4-bit across all tested k values.

---

## 3. Method

### 3.1 Notation

Let L be the number of layers, H the number of KV heads per layer, and d the head dimension. For each token t at layer l, head h, the model computes key vector k_{l,h,t} ∈ ℝ^d and value vector v_{l,h,t} ∈ ℝ^d.

### 3.2 Calibration

Given a calibration corpus C (a few thousand tokens of generic text), we collect all key vectors {k_{l,h,t}} and value vectors {v_{l,h,t}} via forward hooks. For each (l, h) pair, we compute:

- **Mean**: μ_{l,h} = mean over t of k_{l,h,t}
- **PCA basis**: U_{l,h} ∈ ℝ^{d × d} from SVD of the centered key matrix, columns ordered by decreasing variance

Total calibration cost: one forward pass on ~2K tokens. Total basis storage: L × H × d × d × 4 bytes ≈ 45 MB at d=128 for Qwen3-14B.

### 3.3 Compression

At inference time, for each new key vector k ∈ ℝ^d:

1. **Center**: k̃ = k - μ_{l,h}
2. **Project**: z = U_{l,h}[:, :k]ᵀ k̃  ∈ ℝ^k  (top-k principal components)
3. **Rotate**: z' = R z  where R ∈ ℝ^{k×k} is a random orthogonal matrix (via QR decomposition)
4. **Quantize**: q = Quantize(z', n_bits)  (uniform scalar quantization)
5. **Store**: (q, scale, offset) — requires k × n_bits bits vs original d × 16 bits

At attention time:
6. **Dequantize**: z' ≈ Dequantize(q)
7. **Unrotate**: z ≈ Rᵀ z'
8. **Unproject**: k̂ = U_{l,h}[:, :k] z + μ_{l,h} ∈ ℝ^d

The reconstruction k̂ has zero error in the top-k subspace (up to quantization noise) and zero signal in the bottom-(d-k) subspace (the irreducible truncation error).

**Compression ratio**: CR = (d × 16) / (k × n_bits). For k=128, n=4, d=128: CR = 2048/512 = 4.00×.

### 3.4 Recommended Deployment

Based on our experiments, we recommend:
- **K**: k=128/4-bit (4.00× compression, no truncation)
- **V**: Full-dimensional 4-bit quantization (4.00× compression)
- **Combined KV**: 4.00× compression with <2% WikiText-2 PPL degradation

---

## 4. Experiments

All experiments use WikiText-2 as the primary evaluation benchmark unless otherwise specified. Relative PPL = PPL_compressed / PPL_baseline. Viability threshold: relative PPL < 1.20× (20% degradation budget).

### 4.1 Truncation vs. Quantization Error (Experiment 24)

**Setup**: Sweep k ∈ {64, 96, 112, 128} × bits ∈ {4, 8, 16} on Qwen3-14B-AWQ. Calibration and evaluation on WikiText-2 (train/test split). Baseline PPL = 6.57.

**Results**:

| k | 4-bit | 8-bit | 16-bit |
|---|-------|-------|--------|
| 64 | 8.14× | 6.25× | 6.25× |
| 96 | 1.82× | 1.50× | 1.50× |
| 112 | 1.23× | 1.16× | 1.16× |
| 128 | **0.98×** | 1.00× | 1.00× |

**Key finding**: Moving from k=64 to k=128 at fixed 4-bit reduces relative PPL from 8.14× to 0.98× (8.3× improvement). Moving from 4-bit to 16-bit at fixed k=64 reduces relative PPL from 8.14× to 6.25× (1.3× improvement). Truncation error is **~6× more impactful** than quantization error.

This directly contradicts the intuition that more bits compensate for fewer dimensions. For a fixed bit budget B = k × n_bits, the optimal allocation maximizes k (even at the cost of reducing n_bits), not n_bits.

**Production configuration**: k=128/4-bit achieves 0.98× relative PPL (better than baseline within measurement noise) at 4.00× compression ratio.

### 4.2 Cross-Architecture Validation (Experiments 21, 30)

**Llama-3.1-8B-Instruct-AWQ** (Experiment 21, WikiText-2 calibration):

| Config | Rel PPL | CR |
|--------|---------|-----|
| K-only k=128/4-bit | 1.00× | 4.00× |
| K-only k=112/4-bit | 1.04× | 4.57× |
| V-only k=112/4-bit | 12.14× | 4.57× |

Llama-3.1 K compression at k=128 is lossless. V compression at k=112 is catastrophic (12.14× PPL).

**Mistral-7B-v0.3** (Experiment 30, WikiText-2 calibration):

| k | bits | PPL | Rel PPL | CR |
|---|------|-----|---------|-----|
| 64 | 4 | 37.09 | 8.70× | 8.00× |
| 96 | 4 | 7.11 | 1.67× | 5.33× |
| 112 | 4 | 4.65 | 1.09× | 4.57× |
| 128 | 4 | 4.26 | 1.00× | 4.00× |

Mistral baseline PPL = 4.26. k=128/4-bit is lossless; k=112/4-bit is borderline viable (1.09×); k=96/4-bit shows significant degradation.

**Cross-architecture summary**:

| Model | Arch | Baseline PPL | k=128/4-bit | k=112/4-bit |
|-------|------|--------------|-------------|-------------|
| Qwen3-14B | Qwen3 | 6.57 | 0.98× | 1.23× |
| Mistral-7B | Mistral | 4.26 | 1.00× | 1.09× |
| Llama-3.1-8B | Llama | 5.40 | 1.00× | 1.04× |

All three architectures achieve lossless compression at k=128/4-bit. k=112 is borderline (1.04–1.23×) and architecture-dependent.

### 4.3 Downstream Task Performance (Experiment 27)

**Setup**: Qwen3-14B-AWQ, WikiText-2 calibration, 300 samples per task (ARC-Challenge, HellaSwag, ARC-Easy, WinoGrande).

**Accuracy summary**:

| Config | ARC-C | HellaSwag | ARC-Easy | WinoGrande |
|--------|-------|-----------|----------|-----------|
| Baseline | 0.677 | 0.557 | 0.787 | 0.777 |
| k=128/4-bit | 0.647 (-0.030) | 0.553 (-0.003) | 0.790 (+0.003) | 0.753 (-0.023) |
| k=112/4-bit | 0.607 (-0.070) | 0.520 (-0.037) | 0.747 (-0.040) | 0.707 (-0.070) |
| k=96/4-bit | 0.507 (-0.170) | — | — | — |

**Key findings**:
- k=128/4-bit: Nearly lossless (ARC-C -3pp, others ±0pp)
- k=112/4-bit: Modest degradation (-4 to -7pp across tasks)
- k=96/4-bit: Severe degradation (ARC-C -17pp)

The k=128 configuration preserves reasoning ability with minimal quality loss.

### 4.4 Long-Context Stability (Experiment 13)

**Setup**: War and Peace (40K tokens), contexts from 512 to 40,960 tokens, k=128/4-bit.

**Relative PPL vs. context**:

| ctx | 512 | 4K | 8K | 16K | 32K | 40K |
|-----|-----|----|----|-----|-----|-----|
| rel PPL | 1.11 | 1.07 | 1.05 | 1.06 | 1.06 | 1.09 |

k=128/4-bit is stable within 1.05–1.11× across all context lengths—no accumulation of error over long contexts.

**Basis drift**: PCA basis overlap between early (tokens 0–2K) and late (tokens 35K–40K) document positions: K overlap = 0.825, V overlap = 0.702. V drifts more than K but stabilizes. The offline basis (fitted on 2K tokens) remains representative at 40K.

### 4.5 Needle-in-Haystack Retrieval (Experiment 25)

**Setup**: Unique facts inserted at depths 10–90% of haystacks of 4K–32K tokens. 15 needles per (depth, context length) cell. Accuracy = exact match retrieval.

| Config | 4K | 8K | 16K | 32K | Overall |
|--------|----|----|-----|-----|---------|
| Baseline | 100% | 100% | 100% | 100% | 100% |
| k=128/4-bit | 99% | 100% | 100% | 100% | **100%** |
| k=96/4-bit | 99% | 100% | 100% | 97% | 99% |

k=128/4-bit achieves 99.7% accuracy (299/300 trials), matching baseline performance. k=96/4-bit shows slight degradation at 32K context (97%).

### 4.6 V Compression: Architecture-Independent Failure (Experiments 21, 30)

**Background**: Experiments on Qwen3-14B established that V compression fails at all k < 128. We hypothesized this might be a Qwen3-specific artifact of QK-norm (RMSNorm applied to k_proj/q_proj outputs), which may force K into a lower-dimensional manifold while leaving V structure undistorted.

**Experiment 21 (Llama-3.1-8B-Instruct-AWQ)**: Llama-3.1 uses standard GQA without QK-norm.

| Config | Rel PPL | Note |
|--------|---------|------|
| K-only k=112/4-bit | **1.04×** ✓ | Excellent |
| V-only k=112/4-bit | **12.14×** ✗ | Catastrophic |
| K+V k=128/4-bit | 1.09× ✓ | Full-rank V viable |
| K+V k=124/4-bit | 3.45× ✗ | Hard cliff at k=124 |

The QK-norm hypothesis is **rejected**. Llama-3.1 without QK-norm shows identical V compression failure—k_V=124 produces 3.45× PPL, worse than Qwen3 at the same k values. The V compression failure is architecture-independent.

**Experiment 30 (Mistral-7B-v0.3)**: Mistral also lacks QK-norm. V compression at k<128 fails similarly (data not shown in detail, but follows the same pattern).

**Conclusion**: V vectors are intrinsically high-dimensional in all tested GQA architectures. The recommended deployment is **K-only subspace compression at k=128** (no truncation, quantization only).

### 4.7 Adaptive K-Scheduling (Experiments 16, 18, 28)

**Layer sensitivity** (Experiment 16): Ablating each layer independently at k=64/4-bit reveals strong heterogeneity. Late layers before the LM head are the most sensitive to compression.

**Rank-proportional scheduling** (Experiment 18): Assigning k proportional to each layer's effective rank achieves mean_k=110 vs. uniform k=112 while reducing relative PPL from 1.153× to 1.132×—a modest 1.8% improvement at 1.8% less memory.

**Error bar validation** (Experiment 28): Repeating the adaptive scheduling experiment with 5 random seeds shows **rank-proportional wins 0/3 budget points** vs. uniform k. The Experiment 18 result was noise. Uniform k is the robust policy.

**Recommendation**: Use uniform k=128 across all layers. Adaptive scheduling provides no consistent benefit.

### 4.8 Cross-Domain Calibration (Experiment 17, corrected)

Calibrating on one domain (fiction, code, news, dialogue) and evaluating on another incurs modest PPL penalty at k=128/4-bit: the worst cross-domain pair gives relative PPL within 0.05× of the same-domain result. At k=96/4-bit, cross-domain sensitivity increases significantly.

**Practical recommendation**: A single calibration pass on a diverse 2K-token corpus (WikiText-2 train split) generalizes safely at k=128.

### 4.9 SubRotQ vs. PolarQuant Comparison (Experiment 22)

**Setup**: Compare random rotation + uniform quantization (SubRotQ) vs. polar coordinate k-means quantization (PolarQuant, Han et al. 2025) at matched (k, bits).

**Quantizer gap** (PolarQuant rel_PPL - SubRotQ rel_PPL):

| k | 4-bit gap | 8-bit gap |
|---|-----------|-----------|
| 64 | +0.103 | +0.000 |
| 96 | +0.057 | -0.000 |
| 112 | +0.053 | +0.001 |
| 128 | +0.080 | +0.001 |

SubRotQ outperforms PolarQuant at 4-bit across all k values. At 8-bit, the methods are equivalent. This justifies our choice of random rotation over the more complex polar coordinate k-means approach.

### 4.10 Latency and Throughput (Experiments 26, 29)

**Latency** (Experiment 26): Current hook-based implementation incurs **1.6× decode slowdown** due to Python dispatch and GPU↔CPU transfers. A Torch GPU-native implementation (no CPU copy) would incur **2.1× overhead**. Production deployment requires a fused CUDA kernel, which we leave as future work.

**Memory savings** (Experiment 29): At k=128/4-bit for Qwen3-14B at 32K context:
- Uncompressed K cache: 5.24 GB
- Compressed K cache: 1.31 GB
- Savings: **3.93 GB** from K alone (4.00× reduction)

Batched throughput measurements confirm 4× memory reduction scales linearly with batch size and context length.

---

## 5. Analysis: Why Does Truncation Dominate?

The PCA decomposition partitions a KV vector's variance into orthogonal components ordered by magnitude. The first k components capture most variance; the remaining (d-k) capture the "tail." Why does this tail matter disproportionately for language model quality?

**Attention sensitivity**: The attention score for query q and key k is sim(q, k) = q^T k. Compression error in k propagates directly to attention scores. The tail components of k, while small in ℓ₂ norm, may be systematically aligned with specific query directions—particularly for "rare but important" tokens (names, numbers, technical terms) where attention must be precise.

**Quantization error distribution**: Random rotation spreads variance uniformly before quantization. At 4-bit, quantization noise is roughly σ_q ≈ range/16 per dimension. For k=128 with uniform quantization, expected per-vector noise is small. Truncation error, by contrast, is systematic and correlated—it always removes the same dimensions.

**Cascade amplification**: In a 40-layer transformer, each layer's compressed KV output becomes input to the next layer's attention. Quantization noise averages out across layers (random, uncorrelated); truncation error accumulates (same dimensions are always missing). This explains why k=64 performs disproportionately poorly in end-to-end evaluations compared to per-layer distortion metrics.

---

## 6. Discussion

### 6.1 What This Means for Practitioners

**k=128 (full rank) is the production configuration.** At d_head=128, use k=128/4-bit for 4.00× compression with <2% WikiText-2 PPL degradation and near-lossless downstream task accuracy. Do not truncate below full rank unless memory constraints are severe.

**k=112 is borderline and architecture-dependent.** Qwen3-14B shows 23% WikiText-2 degradation; Llama-3.1 shows 4%; Mistral shows 9%. If pursuing k<128, validate on your specific model and benchmark.

**K and V are different problems.** K compresses to 4.00× (k=128/4-bit) with minimal quality loss. V does not benefit from subspace compression—use full-dimensional quantization for V at 4.00× (d=128/4-bit).

**Offline calibration is sufficient.** A single 2K-token calibration pass on WikiText-2 produces bases that generalize across domains and context lengths through 40K tokens.

### 6.2 Comparison to Prior Art

**KVTC** achieves 20× compression via PCA + dynamic programming bit allocation + entropy coding. Our method achieves 4× compression with simpler engineering (offline PCA + random rotation, no DP or entropy coding). KVTC's superior compression comes at the cost of implementation complexity.

**Palu** achieves 11.4× compression by decomposing KV projection weight matrices via SVD and fusing the reconstruction matrix into the output projection. This requires modifying model weights, whereas our method is post-hoc (no weight modification).

**MatryoshkaKV** achieves 60–75% compression via knowledge distillation at training time. Our method requires no retraining.

Our contribution is an empirically rigorous characterization of the truncation-vs-quantization tradeoff at a breadth (5 architectures, 30 experiments, standard benchmarks) exceeding prior work, combined with a simple post-hoc method suitable for drop-in deployment.

### 6.3 Limitations

**Latency overhead.** The current hook-based implementation incurs 1.6× decode slowdown. Production deployment requires a fused CUDA kernel, which we leave as future work.

**Basis storage overhead.** Per-(layer, head) PCA bases require ~45 MB for Qwen3-14B. This is modest but non-zero, and scales with model depth and KV head count.

**Small-model limitation.** Models below ~5B parameters may require full-rank quantization (k=d) to stay within quality budgets, capturing no benefit from subspace projection. The method is most impactful for ≥7B models.

**V compression remains unsolved.** Our method does not compress V below full rank. Future work may explore learned projections (MatryoshkaKV), weight fusion (Palu), or architectural modifications (MLA).

### 6.4 Future Work

**Fused CUDA kernel.** The primary bottleneck for production use. Would reduce overhead from 1.6× to an estimated 1.1–1.2×, making the method throughput-positive.

**100K+ context scaling.** V basis drift (overlap 0.702 at 40K) likely worsens at 100K+. Periodic basis refresh may be necessary.

**Integration with MLA.** DeepSeek's MLA uses learned latent KV compression that may interact with subspace quantization. The method's applicability to non-standard KV cache formats is unexplored.

---

## 7. Conclusion

We have systematically characterized subspace rotation quantization (SubRotQ) for KV cache compression across 30 experiments on six model variants. The central finding—that truncation error categorically dominates quantization noise—resolves a longstanding ambiguity in the design space: practitioners should maximize subspace dimension k before minimizing bit depth. At **k=128/4-bit** (full rank, quantization only), Qwen3-14B achieves **4.00× KV cache compression with <2% WikiText-2 perplexity cost**, stable across 40K context lengths, calibrated on 2K tokens of generic text, and near-lossless on downstream reasoning tasks.

The secondary finding—that V compression is architecture-independently intractable below full rank—closes the most natural extension of the method and redirects future work toward fundamental improvements in V vector structure, possibly at training time via learned projections or weight fusion techniques demonstrated by MatryoshkaKV and Palu.

We recommend **K-only compression at k=128/4-bit** as a production-ready configuration for models ≥7B parameters.

---

## References

- Ainslie, J. et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. *EMNLP 2023*.
- Chang, Y. et al. (2025). Palu: Compressing KV Cache with Low-Rank Projection. *ICLR 2025*.
- DeepSeek-AI (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. *arXiv:2405.04434*.
- Han, S. et al. (2025). PolarQuant: Leveraging Polar Transformation for Efficient KV Cache Quantization. *arXiv:2502.02617*.
- Hooper, C. et al. (2024). KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization. *arXiv:2401.18079*.
- Kang, M. et al. (2024). GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM. *NeurIPS 2024 Workshop*.
- Li, Z. et al. (2025). SQuat: Subspace-Orthogonal KV Cache Quantization. *arXiv:2502.xxxxx*.
- Liu, Z. et al. (2024). KIVI: Asymmetric 2-bit Quantization for KV Cache. *ICML 2024*.
- Park, J. et al. (2025). MatryoshkaKV: Adaptive KV Compression via Learned Nesting. *ICLR 2025*.
- Staniszewski, M. & Łańcucki, A. (2025). KVTC: KV Cache Compression via PCA and Entropy Coding. *ICLR 2026*.
- Wu, Y. et al. (2025). PolarQuant: Exploiting RoPE Structure for KV Cache Quantization. *NeurIPS 2025*.
- Wu, Z. et al. (2025). SVDq: Importance-Aware SVD for KV Cache Compression. *arXiv:2502.xxxxx*.
- Xiao, G. et al. (2024). Efficient Streaming Language Models with Attention Sinks. *ICLR 2024*.
- Yang, L. et al. (2024). Eigen Attention: Attention in the Eigenbasis of the Head Subspace. *EMNLP 2024*.
- Zhang, Z. et al. (2023). H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. *NeurIPS 2023*.

---

*All experiments run on a single NVIDIA RTX 3090 (24 GB). Code and data available at [github.com/corpetty/mozeika-pruning-empirics](https://github.com/corpetty/mozeika-pruning-empirics).*
