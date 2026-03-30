# Subspace PolarQuant: KV Cache Compression via PCA Projection and Rotation Quantization

**Corbin Petty**  
`corpetty@[institution]`

---

## Abstract

We study compression of the key-value (KV) cache in transformer language models via projection into a learned PCA subspace followed by PolarQuant rotation-quantization. Through 21 experiments across five architectures (Qwen3-1.7B/14B/32B, Mistral-7B, Phi-4, Llama-3.1-8B), we establish that **truncation error from dimension reduction categorically dominates quantization noise**: k=64/16-bit (pure truncation) produces 2.48× perplexity degradation while k=128/4-bit (zero truncation, aggressive quantization) produces only 1.05×. This motivates a compression target of k/d_head ≥ 0.875, yielding 4.27× compression at 14% perplexity cost on Qwen3-14B. We further find that **V compression is not viable at any k < d_head across all tested architectures**—including Llama-3.1 which lacks QK-norm—indicating that V vectors are intrinsically high-dimensional, not an artifact of a specific normalization scheme. K compression generalizes across architectures (4× compression, <5% PPL on Llama-3.1; <20% PPL on all tested models at k=112), context lengths (stable to 40K tokens), and calibration domains (single-pass offline calibration transfers across domains). We release `kvpatch`, a library enabling drop-in KV compression via `patch(model, tokenizer, k=112)`.

---

## 1. Introduction

Inference with large language models is increasingly memory-bound at long context lengths. For a 14-billion-parameter model like Qwen3-14B at 32K tokens, the KV cache alone requires approximately 10 GB of GPU memory—comparable to the model weights themselves. This limits the practical context window on consumer hardware and increases serving costs proportionally with sequence length.

KV cache compression aims to reduce this memory footprint at inference time without modifying model weights. The central challenge is that KV vectors must be reconstructed with sufficient fidelity to preserve attention score distributions, which in turn determines the model's output quality.

We investigate a method combining two techniques: **PCA subspace projection**, which projects each d_head-dimensional KV vector into a k-dimensional principal subspace fitted on calibration data, and **PolarQuant** [Han et al., 2025], a learned-rotation quantization scheme that applies an orthogonal rotation before scalar quantization to decorrelate dimensions. The projection reduces the dimensionality to be quantized; PolarQuant reduces quantization error within that subspace.

Our primary contribution is an empirical characterization of this method across scales and architectures, producing three actionable findings:

1. **Truncation error is the binding constraint.** The critical design variable is k (subspace dimension), not bit depth. Practitioners should maximize k before minimizing bits.

2. **V compression is universally hard.** Key vectors compress well across all tested architectures. Value vectors resist subspace compression at any k < d_head, regardless of whether the architecture uses QK-norm. This is an intrinsic property of V variance structure, not an artifact of Qwen3.

3. **K compression is architecture-agnostic and long-context-stable.** K subspace compression generalizes across Qwen3, Mistral, Phi-3, and Llama-3.1, and maintains stable quality through 40K-token contexts with offline calibration.

---

## 2. Background and Related Work

### 2.1 KV Cache Compression

Prior work on KV cache compression falls broadly into three categories: **token eviction** (discarding keys and values for tokens deemed less important [Zhang et al., 2023; Ge et al., 2023]), **quantization** (reducing bit precision of stored KV vectors [Hooper et al., 2024; Liu et al., 2024]), and **dimensionality reduction** (projecting KV vectors to lower-rank representations).

Dimensionality reduction approaches are less studied. GQA [Ainslie et al., 2023] reduces KV head count at training time; our work instead reduces the per-head vector dimensionality at inference time without retraining. SVD-based compression of weight matrices [Hsu et al., 2022] is related but targets static weights rather than dynamic activations.

### 2.2 PolarQuant

PolarQuant [Han et al., 2025] addresses the non-uniform distribution of transformer activations, which makes standard scalar quantization lossy. The key insight is that applying a learned orthogonal rotation R before quantization spreads variance more uniformly across dimensions, reducing per-dimension quantization error. PolarQuant finds R by minimizing quantization error on calibration data. In our implementation, we use PolarQuant as the quantizer within the projected subspace.

### 2.3 PCA Subspace Structure of KV Caches

The KV vectors produced by transformer attention heads exhibit low effective rank. In our analysis of Qwen3-14B, K vectors have a mean effective rank (90% variance threshold) of 29.6 out of d_head=128, while V vectors have mean effective rank 54.2. This asymmetry—K is intrinsically lower-rank than V—motivates asymmetric treatment and foreshadows the fundamental finding that V compression is harder.

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
3. **Rotate**: z' = R z  where R ∈ ℝ^{k×k} is a random rotation matrix
4. **Quantize**: q = Quantize(z', n_bits)  (uniform scalar quantization)
5. **Store**: (q, scale, offset) — requires k × n_bits bits vs original d × 16 bits

At attention time:
6. **Dequantize**: z' ≈ Dequantize(q)
7. **Unrotate**: z ≈ Rᵀ z'
8. **Unproject**: k̂ = U_{l,h}[:, :k] z + μ_{l,h} ∈ ℝ^d

The reconstruction k̂ has zero error in the top-k subspace (up to quantization noise) and zero signal in the bottom-(d-k) subspace (the irreducible truncation error).

**Compression ratio**: CR = (d × 16) / (k × n_bits). For k=112, n=4, d=128: CR = 2048/448 = 4.57×.

### 3.4 V Treatment

Our experiments establish that V compression fails at k < d for all tested architectures. The recommended deployment therefore uses:
- **K**: Subspace PolarQuant (k=112, 4-bit) — 4.57× compression
- **V**: Full-dimensional PolarQuant (k=d=128, 4-bit) — 4.00× compression
- **Combined KV**: (K saving + V saving) / 2 ≈ 4.27× compression

---

## 4. Experiments

All experiments use a fixed evaluation protocol: three passages (biology, computer science, history) with perplexity as the primary metric. Relative PPL = PPL_compressed / PPL_baseline. Viability threshold: relative PPL < 1.20× (20% degradation budget).

### 4.1 Truncation vs. Quantization Error (Experiment 9)

**Setup**: Sweep k ∈ {64, 96, 112, 128} × bits ∈ {4, 6, 8, 16} on Qwen3-14B-AWQ. Model: Qwen3-14B-AWQ (40 layers, 8 KV heads, d_head=128).

**Results**:

| k | 4-bit | 8-bit | 16-bit |
|---|-------|-------|--------|
| 64 | 3.19× | 2.68× | 2.48× |
| 96 | 1.26× | 1.15× | 1.11× |
| 112 | 1.14× | 1.07× | 1.06× |
| 128 | 1.05× | 1.02× | 1.01× |

**Key finding**: Moving from k=64 to k=128 at fixed 4-bit reduces relative PPL from 3.19× to 1.05× (3.0× improvement). Moving from 4-bit to 16-bit at fixed k=64 reduces relative PPL from 3.19× to 2.48× (0.28× improvement). Truncation error is **~10× more impactful** than quantization error.

This directly contradicts the intuition that more bits compensate for fewer dimensions. For a fixed bit budget B = k × n_bits, the optimal allocation maximizes k (even at the cost of reducing n_bits), not n_bits.

### 4.2 End-to-End Perplexity at Scale (Experiments 1–12)

**Qwen3 family scaling** (k=112/4-bit):

| Model | Params | Rel PPL | CR |
|-------|--------|---------|-----|
| Qwen3-1.7B | 1.7B | 1.32× | 4.27× |
| Qwen3-14B-AWQ | 14B | 1.14× | 4.27× |
| Qwen3-32B-AWQ | 32B | 1.06× | 4.27× |

Compression tolerance scales with model size within an architecture family. The 32B model tolerates k=96 at 1.09× PPL (4.57× CR), while the 1.7B model requires full dimensions.

**Cross-architecture** (k/d_head = 0.875 = k=112 for d=128):

| Model | Arch | Rel PPL (k=112) | Min viable k | CR at min-k |
|-------|------|-----------------|--------------|-------------|
| Mistral-7B-v0.3 | Mistral | 1.07× | 64 | 5.33× |
| Phi-4-AWQ | Phi3 | 1.10× | 64 | 5.33× |
| Qwen3-14B-AWQ | Qwen3 | 1.14× | 112 | 4.27× |
| Llama-3.1-8B | LLaMA | 1.04× | 112* | 4.57× |

*Llama-3.1 K-only (V excluded); see §4.5.

Mistral and Phi-3 architectures tolerate k=64 (50% truncation) within the 20% budget — more than twice as aggressive as Qwen3. Llama-3.1 achieves the best K-only result (1.04×) of any tested architecture.

### 4.3 Long-Context Stability (Experiment 13)

**Setup**: War and Peace (40K tokens), contexts from 512 to 40,960 tokens, sub-experiments measuring (A) PPL vs. context length, (B) per-position error distribution, (C) PCA basis drift.

**Relative PPL vs. context** (k128/4-bit):

| ctx | 512 | 4K | 8K | 16K | 32K | 40K |
|-----|-----|----|----|-----|-----|-----|
| rel PPL | 1.11 | 1.07 | 1.05 | 1.06 | 1.06 | 1.09 |

k128/4-bit is stable within 1.05–1.11× across all context lengths—no accumulation of error over long contexts. Per-position analysis (sub-exp B) confirms compression errors are uniformly distributed across token positions, with no late-sequence blowup.

**Basis drift** (sub-exp C): PCA basis overlap between early (tokens 0–2K) and late (tokens 35K–40K) document positions: K overlap = 0.825, V overlap = 0.702. V drifts more than K but stabilizes. The offline basis (fitted on 2K tokens) remains representative at 40K.

### 4.4 Task Performance: Needle-in-Haystack Retrieval (Experiment 15)

**Setup**: Unique facts inserted at depths 10–90% of haystacks of 4K–32K tokens. Accuracy = exact match retrieval of the fact.

| Config | 4K | 8K | 16K | 32K | Overall |
|--------|----|----|-----|-----|---------|
| Baseline | 93% | 93% | 87% | 100% | 93% |
| k128/4-bit | 93% | 93% | 100% | 100% | **97%** |
| k96/4-bit | 100% | 93% | 100% | 27% | 80% |

k128/4-bit matches or exceeds baseline accuracy across all context lengths, achieving 97% overall vs. 93% baseline. This surprising improvement at 16K and 32K is likely a mild regularization effect — slight compression noise reducing overconfident attention patterns.

k96/4-bit collapses at 32K (27% accuracy), consistent with its PPL instability above 16K context.

### 4.5 V Compression: Architecture-Independent Failure (Experiments 19–21)

**Background**: Experiments 19–20 established that V compression fails for Qwen3-14B at all k < 128. Experiment 20 showed k_V=128 (full rank, 4-bit quantization only) is borderline viable at 4K context but fails at 8K. We hypothesized this might be a Qwen3-specific artifact of QK-norm (RMSNorm applied to k_proj/q_proj outputs, which may force K into a lower-dimensional manifold while leaving V structure undistorted).

**Experiment 21 (Llama-3.1-8B-Instruct-AWQ)**: Llama-3.1 uses standard GQA without QK-norm.

| Config | Rel PPL | Note |
|--------|---------|------|
| K-only k=112/4-bit | **1.042×** ✓ | Excellent — best K result across all architectures |
| V-only k=112/4-bit | **12.14×** ✗ | Catastrophic — worse than Qwen3 at same k |
| K+V k=128/4-bit | 1.085× ✓ | Full-rank V quantization viable |
| K+V k=124/4-bit | 3.45× ✗ | Hard cliff at k=124 |

The QK-norm hypothesis is **rejected**. Llama-3.1 without QK-norm shows identical V compression failure — k_V=124 produces 3.45× PPL, k_V=120 produces 6.68× PPL, catastrophically worse than Qwen3 at the same k values. The V compression failure is architecture-independent.

**Conclusion**: V vectors are intrinsically high-dimensional in all tested GQA architectures. The ~30% of V variance residing in tail principal components (dimensions 113–128 for d_head=128) is load-bearing signal, not quantization noise. The recommended deployment is K-only subspace compression.

### 4.6 Adaptive K-Scheduling (Experiments 16, 18)

**Layer sensitivity** (Experiment 16): Ablating each layer independently at k=64/4-bit reveals strong heterogeneity. Layer 37 is most sensitive (+1.93 PPL delta), while layers 2, 20, 25, 27 show *negative* sensitivity (slight improvement under compression — mild regularization). Late layers before the LM head are the most sensitive.

**Rank-proportional scheduling** (Experiment 18): Assigning k proportional to each layer's effective rank achieves mean_k=110 vs. uniform k=112 while reducing relative PPL from 1.153× to 1.132× — a free 1.8% improvement at 1.8% less memory. This policy is straightforward to compute from the calibration forward pass.

### 4.7 Cross-Domain Calibration (Experiment 17)

Calibrating on one domain (fiction, code, news, dialogue) and evaluating on another incurs modest PPL penalty at k=128/4-bit: the worst cross-domain pair gives relative PPL within 0.05× of the same-domain result. At k=96/4-bit, cross-domain sensitivity increases: code→news gives relative PPL 2.70× vs. universal-calibrated 1.63×.

**Practical recommendation**: A single calibration pass on a diverse 2K-token corpus generalizes safely at k=128. For more aggressive k=96, universal calibration (mixed-domain) is required.

---

## 5. Analysis: Why Does Truncation Dominate?

The PCA decomposition partitions a KV vector's variance into orthogonal components ordered by magnitude. The first k components capture most variance; the remaining (d-k) capture the "tail." Why does this tail matter disproportionately for language model quality?

**Attention sensitivity**: The attention score for query q and key k is sim(q, k) = q^T k. Compression error in k propagates directly to attention scores. The tail components of k, while small in ℓ₂ norm, may be systematically aligned with specific query directions — particularly for "rare but important" tokens (names, numbers, technical terms) where attention must be precise.

**Quantization error distribution**: PolarQuant rotation spreads variance uniformly before quantization. At 4-bit, quantization noise is roughly σ_q ≈ range/16 per dimension. For k=128 with uniform quantization, expected per-vector noise is small. Truncation error, by contrast, is systematic and correlated — it always removes the same dimensions.

**Cascade amplification**: In a 40-layer transformer, each layer's compressed KV output becomes input to the next layer's attention. Quantization noise averages out across layers (random, uncorrelated); truncation error accumulates (same dimensions are always missing). This explains why k=64 performs disproportionately poorly in end-to-end evaluations (Experiments 6–8) compared to per-layer distortion metrics.

---

## 6. The `kvpatch` Library

We release `kvpatch`, a Python library enabling drop-in KV compression for HuggingFace and AWQ models via forward hooks.

### 6.1 API

```python
from kvpatch import patch, KVBasis

# One-call usage: auto-calibrate and patch
patch(model, tokenizer, k=112, bits=4)

# Or with explicit calibration for basis reuse
basis = calibrate(model, tokenizer, k=112, bits=4,
                  save_path="basis.pkl")
patch(model, basis=basis)

# Remove compression
from kvpatch import unpatch
unpatch(model)
```

### 6.2 Architecture Support

`kvpatch` auto-detects architectures by inspecting the model config and module structure: Qwen2/Qwen3 (model.model.model for AWQ, model.model for standard), LLaMA/Mistral (model.model.layers[i].self_attn), Phi-3, Falcon, and a generic fallback scanning `named_modules` for `k_proj`/`v_proj`.

### 6.3 Calibration

Default calibration: one forward pass of 2K tokens on a built-in mixed-domain corpus (biology, CS, history). Custom calibration text can be provided. Basis objects are serializable for reuse across sessions, eliminating re-calibration overhead.

### 6.4 Memory Impact

At k=112, 4-bit, for Qwen3-14B at 32K context:
- Uncompressed K cache: 5.24 GB
- Compressed K cache: 1.15 GB
- Savings: **4.09 GB** from K alone

This enables running 32K context on a single 24GB GPU that would otherwise require splitting across two GPUs or reducing context length.

---

## 7. Discussion

### 7.1 What This Means for Practitioners

**k, not bits, is the design variable.** When tuning compression, increase k first. Going from k=112 to k=128 improves PPL more than doubling bit depth from 4 to 8 at the same k.

**K and V are different problems.** K compresses to 4.57× (k=112/4-bit) with modest quality loss. V does not benefit from subspace compression — use full-dimensional quantization for V at 4.00× (d=128/4-bit). The combined KV compression ratio is approximately 4.27×.

**Offline calibration is sufficient.** A single 2K-token calibration pass on generic text produces bases that generalize across domains, architectures, and context lengths through 40K tokens.

**Large models compress better.** Within the Qwen3 family, compression tolerance increases monotonically with scale. Qwen3-32B achieves k=96 within 10% PPL; Qwen3-1.7B requires full dimensions. This reflects the over-parameterization hypothesis: larger models develop lower-rank functional KV subspaces.

### 7.2 Limitations

**Hook-based implementation overhead.** The current `kvpatch` implementation uses PyTorch forward hooks with CPU-side numpy operations. This incurs ~13× decode slowdown compared to unpatched inference — real throughput benefit requires a fused CUDA kernel for project-rotate-quantize.

**Basis storage overhead.** Per-(layer, head) PCA bases require ~45 MB for Qwen3-14B. This is modest but non-zero, and scales with model depth and KV head count.

**Small-model limitation.** Models below ~5B parameters (Qwen3-1.7B in our tests) require full-rank quantization (k=d) to stay within the 20% PPL budget, capturing no benefit from subspace projection. The method is most impactful for ≥7B models.

### 7.3 Future Work

**Fused CUDA kernel.** The primary bottleneck for production use. Would reduce hook overhead from ~13× to ~1.2× estimated, making the method throughput-positive at batch sizes ≥ 4.

**100K+ context scaling.** V basis drift (overlap 0.702 at 40K) likely worsens at 100K+. A periodic basis refresh strategy (without the online update overhead identified in Exp 19 as futile for improving quality) may be necessary.

**Integration with GQA and MLA.** Modern architectures like DeepSeek's MLA [DeepSeek-AI, 2024] use latent KV compression that may interact with subspace quantization. The method's applicability to non-standard KV cache formats is unexplored.

---

## 8. Conclusion

We have systematically characterized subspace PolarQuant KV compression across 21 experiments on six model variants. The central finding—that truncation error categorically dominates quantization noise—resolves a longstanding ambiguity in the design space of KV cache compression: practitioners should maximize subspace dimension k before minimizing bit depth. At k=112/4-bit, Qwen3-14B achieves 4.27× KV cache compression with 14% perplexity cost, stable across 40K context lengths, calibrated on two thousand tokens of generic text, and applicable via three lines of Python code.

The secondary finding—that V compression is architecture-independently intractable below full rank—closes the most natural extension of the method and redirects future work toward fundamental improvements in V vector structure, possibly at training time via V-norm analogues to QK-norm.

---

## References

- Ainslie, J. et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. *EMNLP 2023*.
- DeepSeek-AI (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. *arXiv:2405.04434*.
- Ge, S. et al. (2023). Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs. *arXiv:2310.01801*.
- Han, S. et al. (2025). PolarQuant: Leveraging Polar Transformation for Efficient KV Cache Quantization. *arXiv:2502.02617*.
- Hooper, C. et al. (2024). KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization. *arXiv:2401.18079*.
- Hsu, Y. et al. (2022). Language Model Compression with Weighted Low Rank Factorization. *ICLR 2022*.
- Liu, Z. et al. (2024). KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing. *arXiv:2407.00519*.
- Zhang, Z. et al. (2023). H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. *NeurIPS 2023*.

---

*All experiments run on a single NVIDIA RTX 3090 (24 GB) unless otherwise noted. Code and data available at [github.com/corpetty/mozeika-pruning-empirics](https://github.com/corpetty/mozeika-pruning-empirics).*
