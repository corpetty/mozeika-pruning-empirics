# What We're Actually Doing Here: KV Cache Compression

*For people who want to understand the research without needing a PhD in ML.*

---

## The Big Picture

When a large language model generates text, it doesn't just look at the current word — it looks at *every previous word in the conversation*. To do this efficiently, it stores a compact representation of every token it has seen, called the **KV cache** (Key-Value cache). The problem: this cache grows linearly with context length and eats GPU memory at an alarming rate.

For Qwen3-14B with a 4,096-token context window, the KV cache consumes about **168 MB in FP16**. That's just for one conversation. Run a few long sessions in parallel, scale to production, and the KV cache becomes the binding memory constraint — not the model weights themselves.

**The question we're answering:** Can we compress the KV cache aggressively (4× or more) without meaningfully degrading the model's output quality?

---

## How the KV Cache Works

Every attention head in every layer of a transformer stores two 128-dimensional vectors per token: a **key** (K) and a **value** (V). You can think of these as:
- **Keys:** What this token is offering to be found by
- **Values:** What information this token passes along when found

When the model generates the next token, it computes a similarity (attention score) between the current query and all stored keys, then uses those scores to mix together the corresponding values. The output is a weighted combination of all the information seen so far.

This is powerful but expensive. At 128 dimensions per vector, 2 vectors (K and V) per head, 8 heads per layer, 40 layers, and 4096 tokens, you get: 128 × 2 × 8 × 40 × 4096 × 2 bytes (FP16) = 168 MB. Every token you add increases this linearly.

---

## Our Approach: Subspace Projection + Quantization

We combine two complementary compression techniques:

### Step 1: PCA Subspace Projection

Principal Component Analysis (PCA) finds the directions in which your data varies the most. Think of a cloud of points in 3D space that's mostly flat — PCA would find the two directions that capture 95% of the spread, letting you describe each point with 2 numbers instead of 3.

KV vectors turn out to have dramatically lower *effective dimensionality* than their nominal 128 dimensions suggest. We ran PCA on the KV vectors of a real forward pass and found:
- **Key vectors:** On average, only 30 dimensions are needed to capture 90% of the variance
- **Value vectors:** Higher — about 54 dimensions on average

So instead of storing 128 numbers per vector, we:
1. **Fit a PCA basis offline** (one calibration run on a sample of text)
2. **At inference time:** project each 128-dim KV vector down to k dims (we found k=112 to be the sweet spot)
3. Store only the k-dimensional projection

This is lossy — the bottom (128-k) dimensions are permanently discarded. As we'll see, *how* lossy matters enormously.

### Step 2: PolarQuant

The projected k-dimensional vectors are still floating-point numbers. **PolarQuant** compresses them further using scalar quantization (representing each number with fewer bits — e.g., 4 bits instead of 16).

The trick PolarQuant uses: before quantizing, apply a learned rotation matrix that decorrelates the dimensions. Correlated dimensions are hard to quantize efficiently; uncorrelated ones can each be quantized independently without much loss. The rotation makes quantization significantly more accurate at the same bit count.

### Combined compression ratio

For k=112 dimensions at 4 bits per dimension:
- **Original:** 128 dims × 16 bits = 2048 bits per vector
- **Compressed:** 112 dims × 4 bits = 448 bits per vector
- **Compression ratio: 4.27×**

For a 4K context with 40 layers and 8 heads, this takes the 168 MB KV cache down to about **39 MB**.

---

## The Critical Finding: Truncation Beats Quantization

Before running end-to-end tests, we thought the main challenge would be quantization noise. It wasn't.

We ran a systematic sweep varying both the subspace dimension k and the bit depth:

| Config | Description | Perplexity Impact |
|--------|-------------|-------------------|
| k=64 / 16-bit | Half dimensions, lossless quantization | **2.48× worse** |
| k=128 / 4-bit | Full dimensions, aggressive quantization | **1.05× worse** |
| k=112 / 4-bit | Sweet spot | **1.14× worse** |
| k=64 / 4-bit | Initial naive guess | **3.19× worse** |

The smoking gun: **truncation error is 24× more damaging than quantization error.** Dropping from 128 to 64 dimensions while keeping full precision causes nearly as much damage as compression can cause — and adding bits back doesn't help, because the dropped dimensions are gone forever.

This is actually a classical result from information theory: rate-distortion theory says dimension reduction (source coding) hurts more than quantization (channel coding), because quantization noise can be made small with more bits, but truncated dimensions are irrecoverable. We rediscovered this empirically in transformer attention caches.

**The practical lesson:** Don't be aggressive about how many dimensions you keep. Keep ≥87.5% of them (k/d_head ≥ 0.875). Then quantize the retained dimensions aggressively. This is the opposite intuition from what most people would guess.

---

## What Perplexity Means

**Perplexity** (PPL) measures how surprised a language model is by text — lower is better. A model with baseline perplexity 2.58 is quite confident about what comes next. If compression raises this to 8.25 (3.19×), the model is suddenly much more "confused," which manifests as worse text quality, more factual errors, and less coherent reasoning.

Our end-to-end tests (on Qwen3-14B-AWQ) found:
- **k=64 / 4-bit:** 8.25 PPL (3.19× worse) — clearly too aggressive
- **k=112 / 4-bit:** 2.95 PPL (1.14× worse) — acceptable
- **k=128 / 4-bit:** 2.72 PPL (1.05× worse) — barely perceptible

The 1.14× PPL increase at k=112 is roughly the difference between a model seeing 10% less context than it was trained on — minor, not catastrophic.

---

## K and V Are Not the Same

An important asymmetry emerged from our effective-rank analysis:

**Key vectors (K):** Mean effective rank ≈ 30. Most of the "signal" is concentrated in the top 30 principal components. Subspace projection works excellently here — k=64 already captures >99% of K variance.

**Value vectors (V):** Mean effective rank ≈ 54. More spread out. At k=64, you're only capturing ~90% of V variance — that missing 10% turns out to matter.

**The practical recommendation:** Apply subspace projection to K, but use full-dimensional quantization for V. The K projection gives you most of the compression benefit; the V projection introduces disproportionate error for the compression gained.

Final recommended pipeline:
- **K:** project to k=112 dims, rotate, quantize at 4-bit → 448 bits/vector
- **V:** full 128 dims, rotate, quantize at 4-bit → 512 bits/vector
- Combined compression ratio: **4.27× vs FP16**

---

## The Cascade Problem (and Why It Disappeared)

Our first end-to-end experiments (k=64) showed a puzzling amplification effect: the per-layer distortion looked manageable, but the final perplexity was catastrophic. Why?

Each transformer layer feeds into the next. A small compression error in layer 5's attention output becomes a slightly corrupted input to layer 6, which produces a more corrupted output to layer 7, and so on for all 40 layers. Small per-layer errors *cascade*. Attention fidelity at early layers (top-1 match ≈ 63%) degraded to only 26% fidelity in late layers.

The intuitive fix — protect early layers from compression, only compress later ones — worked partially but required leaving so many layers uncompressed that the overall compression ratio collapsed from 5.33× to under 2×.

The actual fix was simply increasing k. At k=112, the per-layer error is small enough that 40-layer accumulation stays within budget. The cascade effect is a symptom of too-aggressive truncation, not an independent problem requiring special treatment.

---

## Does the Per-Head Subspace Structure Matter?

We tested whether KV cache subspaces from different attention heads, layers, and text domains are similar enough to share a single projection basis (the Universal Weight Subspace Hypothesis — UWSH).

Results:
- **Cross-domain overlap (K): 0.70** — Fiction and factual text produce similar K subspaces. Good news: one offline calibration run generalizes across domains.
- **Cross-layer overlap (K): 0.56** — Moderate. Early layers (L0-9) are outliers; late layers converge.
- **Cross-head overlap (K): 0.46** — Low. Each attention head learns a distinct subspace.

**Practical implication:** You can't share one projection matrix across all heads. Each of the 8 KV heads per layer needs its own PCA basis. The total storage for all bases is ~45 MB at k=112 — a fixed overhead that's small relative to the per-token KV cache savings at long context.

---

## Offline Calibration: Does It Transfer?

A realistic deployment would fit PCA bases once offline (on a sample of generic text) and apply them to arbitrary user inputs. We tested this explicitly by fitting on fiction text and evaluating on Wikipedia-style factual text.

- **At 2-bit:** Cross-domain transfer is viable — 2× KL penalty versus an oracle (same-domain) basis, but still better than full-dimensional quantization on 99.7% of heads.
- **At 4-bit:** Full-dimensional quantization wins on 91% of heads. Transfer penalty grows to 3.8× KL.

Since our recommended operating point is 4-bit, and we recommend full-dim for V anyway, calibration transfer is a non-issue for the K projection (which is most valuable at 2-bit) and not needed for V. **A single one-time calibration run suffices.**

---

## How Generalizable Is This? (Cross-Model Results)

We tested whether the k/d_head ≥ 0.875 rule generalizes across model sizes and architectures. The results were surprising.

**Within the Qwen3 family:** Larger models tolerate more aggressive truncation.

| Model | Min k/d_head for <20% PPL | Example |
|-------|--------------------------|---------|
| Qwen3-1.7B | 1.0 (full-dim) | Even k=112 causes 32% PPL hit |
| Qwen3-14B-AWQ | 0.875 | k=112: 1.14× PPL ✓ |
| Qwen3-32B-AWQ | 0.75 | k=96: 1.09× PPL ✓ |

**Across architectures (surprising):** Mistral-7B and Phi-4 are dramatically more compression-tolerant than Qwen3 at comparable sizes.

| Model | Architecture | k=64 (half dims) | k=112 |
|-------|-------------|------------------|-------|
| Qwen3-14B-AWQ | Qwen3 | 3.19× PPL ❌ | 1.14× PPL ✓ |
| Phi-4-AWQ | Phi3 | 1.18× PPL ✓ | 1.10× PPL ✓ |
| Mistral-7B-v0.3 | Mistral | 1.12× PPL ✓ | 1.07× PPL ✓ |

Mistral-7B at k=64 outperforms Qwen3-32B at k=64. Architecture matters as much as size. The likely explanation: Mistral and Phi3 attention implementations produce genuinely lower-rank KV vectors — their effective rank is lower, so keeping half the dimensions loses less.

**Architecture-specific recommendations:**
- **Mistral / Phi3:** k=64 works (5.33× compression, <20% PPL)
- **Qwen3 ≥10B:** k=112 recommended (4.27× compression)
- **Qwen3 <5B:** full-dim PolarQuant only (4× compression, no projection)

---

## Hardware Reality

The projection step adds latency. On an RTX 3090 at T=512 tokens:
- **Plain quantization:** ~185 μs/head
- **Subspace PolarQuant (k=112):** ~325 μs/head — **1.7× overhead**

Across 40 layers, this is roughly **5.6 ms per token** of extra latency. For batch/long-context inference (where memory savings dominate), this is acceptable. For real-time single-user chat, it's marginal.

A fused CUDA kernel (combining the projection, rotation, and quantization into one pass) would reduce this to ~1.2×. That's the obvious next engineering step.

---

## Recommended Configurations

### Qwen3-14B-AWQ (the primary model we tested)

| Priority | K strategy | V strategy | PPL impact | Compression |
|----------|-----------|------------|------------|-------------|
| Safety-first | Full-dim 4-bit | Full-dim 4-bit | 1.05× | 4.00× |
| Balanced | k=128 L0-19, k=112 L20-39 | Full-dim 4-bit | 1.10× | 4.13× |
| Max compression | k=112 / 4-bit | Full-dim 4-bit | 1.14× | 4.27× |

### Other architectures

| Model | Architecture | Recommended k | Rel PPL | Compression |
|-------|-------------|--------------|---------|-------------|
| Qwen3-1.7B | Qwen3 | 128 (full-dim) | ~1.13× | 4.00× |
| Mistral-7B-v0.3 | Mistral | 64 | ~1.12× | 5.33× |
| Qwen3-14B-AWQ | Qwen3 | 112 | ~1.14× | 4.27× |
| Phi-4-AWQ | Phi3 | 64 | ~1.18× | 5.33× |
| Qwen3-32B-AWQ | Qwen3 | 96 | ~1.09× | 4.57× |

---

## Honest Limitations

**The truncation floor is real.** Even at full floating-point precision, discarding 13% of dimensions (k=112) costs 1.05× PPL on its own. There is no quantization trick that recovers lost dimensions.

**The hardware cost isn't free.** 1.7× overhead is manageable but not negligible. Until fused kernels exist, this is a real deployment concern.

**We tested at 512-token evaluation sequences.** The compression was calibrated on 2K-token forward passes. Very long contexts (100K+ tokens) may reveal phenomena we haven't seen — e.g., PCA basis drift over very long sequences.

**Calibration matters more at aggressive k.** At k=64 (Mistral/Phi3 operating point), using a bad calibration basis could significantly hurt performance. The 0.70 cross-domain overlap score means you're not completely safe with a random calibration set.

**We didn't test fine-tuned models.** Task-specific fine-tuning may change the KV subspace structure. The effective ranks and recommended k values were measured on base models; fine-tuned variants should be re-evaluated.

---

## What's Next

1. **Fused CUDA kernel:** Combine project-rotate-quantize into one pass, targeting ≤1.2× overhead.
2. **Token-level adaptive compression:** High-attention tokens (beginning of sentence, special tokens, key facts) may warrant lossless or near-lossless storage; low-attention tokens could go more aggressive.
3. **Non-uniform bit allocation:** Late layers have higher effective rank and worse compression tolerance. Giving them more bits (6-bit) while keeping early layers at 2-bit could improve the Pareto frontier.
4. **Testing at 100K+ contexts:** Where KV cache memory savings would actually be most impactful.
5. **Learned projection bases:** PCA optimizes for reconstruction MSE, not attention fidelity. A task-aware basis trained to minimize attention KL directly could achieve the same PPL at lower k.

---

## Why This Fits the Bigger Picture

This research is the inference-time counterpart to the pruning work in the parent directory. Both are asking the same question from different angles: **how much of a neural network's apparent complexity is actually necessary?**

The pruning work asks: which *weights* in a trained model can be removed permanently?
This work asks: which *directions* in the KV cache vector space can be discarded at runtime?

Both find the same answer: much less than you'd think, with a sharp threshold beyond which quality collapses. Both find that the naive approach (prune the bottom X%, truncate to k dimensions) underperforms methods that are geometry-aware (Fisher information, PCA basis + PolarQuant). And both find that the right amount of compression is architecture-dependent in ways that aren't yet fully theoretically understood.

The compression-tolerant architectures (Mistral, Phi3) and the pruning-tolerant architectures (large overparameterized models) may be the same underlying phenomenon: redundancy in learned representations. Understanding that redundancy — where it comes from, why some architectures have more of it than others — is the deeper research question both threads are pointing toward.

---

*Document maintained by Nick Molty 🦝. Last updated: 2026-03-28.*
*Technical details: see RESULTS.md, results/SUMMARY.md, and results/REPORT-1 through REPORT-12.*
*Data and scripts: experiments/ directory.*
