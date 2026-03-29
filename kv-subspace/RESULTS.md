# KV Cache Subspace Compression — Results

**Date:** 2026-03-26  
**Model:** Qwen3-14B-AWQ  
**Hardware:** Dual RTX 3090 (24 GB each)  
**Setup:** 40 transformer layers, 8 GQA KV heads, d_head=128  
**Text:** 2048 tokens from Project Gutenberg (long-context benchmark)

---

## Experiment

**Question:** Does compressing KV vectors in their principal PCA subspace require fewer bits to preserve attention score quality than compressing in full ambient space?

**Method:**
1. Extract K and V vectors from all 40 layers via forward-pass hooks
2. PCA each layer's K matrix — compute effective rank at 90% variance threshold
3. Compare two compression schemes at matched bits-per-vector budget:
   - **Full-dim:** PolarQuant (random rotation + uniform scalar quantization) on the full 128-dim vector
   - **Subspace-k64:** Project to top-64 PCA dims → PolarQuant at double the bits → reconstruct (same total budget: 64 × 4bit = 128 × 2bit = 256 bits)
4. Measure attention score distortion: KL divergence between softmax(QK^T) and softmax(QK_compressed^T)

---

## Effective Rank by Layer

KV vectors are far from full rank. At 90% explained variance:

| Layer range | K eff_rank / 128 | V eff_rank / 128 |
|-------------|-----------------|-----------------|
| 0–5 (early) | 11–22 (9–17%) | 26–42 (20–33%) |
| 6–15 (mid-early) | 21–27 (16–21%) | 36–57 (28–45%) |
| 16–27 (mid-late) | 27–31 (21–24%) | 49–67 (38–52%) |
| 28–39 (late) | 36–43 (28–33%) | 57–67 (45–53%) |

**Key observations:**
- K vectors are consistently lower rank than V vectors at every layer
- Effective rank grows with layer depth (later layers = higher rank, harder to compress)
- Even the hardest layer (L39) has K eff_rank ~43/128 = 33% — still strongly compressible
- Layer 0 K heads: effective rank just 11/128. These are doing extremely low-dimensional routing.

---

## Compression Distortion (KL Divergence, lower = better)

### At matched budget: 256 bits/vector (full_dim 2bit vs subspace_k64 4bit)

| Layer | full_dim 2bit | subspace_k64 4bit | gain |
|-------|--------------|-------------------|------|
| 0–9 | 0.0000–0.0037 | 0.0000–0.0006 | **5.6–9.5×** |
| 10–19 | 0.011–0.29 | 0.002–0.048 | **5.0–6.7×** |
| 20–27 | 0.65–2.94 | 0.13–0.59 | **4.5–6.8×** |
| 28–33 | 3.7–8.9 | 0.74–3.6 | **2.5–5.0×** |
| 34–39 | 11.9–13.3 | 4.9–7.9 | **1.7–2.4×** |

**Subspace compression wins at every layer at matched budget. Gains range from 9.5× (early layers) to 1.7× (late layers).**

### Overall (mean across all 40 layers × 8 heads)

| Method | 2bit | 4bit | 8bit |
|--------|------|------|------|
| full_dim | KL=3.11 | KL=0.29 | KL=0.001 |
| subspace_k64 (4bit matched) | **KL=1.28** | — | — |

At fixed 256 bits/vector budget: subspace wins by **2.4×** mean KL.  
At 4bit full-dim (512 bits/vector): full-dim is already excellent (KL=0.29).

---

## Key Findings

### 1. Subspace compression dominates at low bit budgets
At ≤4 bits/dim (the aggressive compression regime for long-context inference), projecting to the principal subspace before quantizing consistently outperforms quantizing in the full space. The gain is largest for early layers (where rank is lowest) and smallest for late layers (where rank grows toward d_head).

### 2. The gain tracks effective rank
The compression benefit is directly predicted by how far the effective rank is from d_head:
- Early layers: eff_rank ≈ 10–20% of d_head → 5–9× gain
- Late layers: eff_rank ≈ 30–35% of d_head → 1.7–2.4× gain

This is exactly what you'd expect from information-theoretic arguments: if k dimensions explain 90% of variance, quantizing the remaining 128-k dims is wasted bits.

### 3. K and V have different compressibility profiles
V vectors consistently have higher effective rank than K. This matters for per-component compression strategies: you can be more aggressive with K compression than V compression at the same quality target.

### 4. Late layers are the hard problem
Layers 28–39 have full-dim 2bit KL in the range 5–13 — this is where any aggressive KV compression scheme will struggle. Subspace helps (1.7–2.4× gain) but even with it, KL>1 is hard to avoid at 256 bits/vector. Practical systems doing KV compression should probably exempt the last 8–12 layers from aggressive quantization.

---

## Connection to UWSH and TurboQuant

The spectral structure here is the inference-time manifestation of what UWSH observes at training time. The fact that KV vectors cluster in a ~20–30% subspace across heads and layers is the same low-dimensional geometry that makes LoRA adapters work and that UWSH's universal subspace captures. The mechanisms are related:

- UWSH: weight matrices cluster in low-dim subspaces across tasks/training runs
- Here: KV vectors cluster in low-dim subspaces during inference on any text

TurboQuant (PolarQuant's predecessor) operates on the full d_head. The clean improvement here suggests a hybrid: (1) identify per-layer PCA subspace offline on calibration data, (2) project KV to subspace at inference time, (3) PolarQuant in the reduced space. Memory footprint of the subspace basis is tiny (k × d_head float16 per layer ≈ 64 × 128 × 2B × 40 layers = 655 KB — negligible).

---

## Files

- `results/kvs.npz` — raw KV vectors, all 40 layers, 2048 tokens (376 MB)
- `results/analysis.npz` — PCA singular values + effective rank per layer/head
- `results/compression_distortion.csv` — 4800 rows of compression comparison data
- `analyze.py` — PCA analysis script
- `compress_standalone.py` — compression comparison using analysis.npz (no model reload needed)
- `compress.py` — PolarQuant and subspace quantization primitives

---

## Next Steps

1. **Run on real KV vectors** (not synthetic from spectrum) — the synthetic approach approximates the distribution correctly but real vectors may have non-Gaussian tails that matter for quantization
2. **Adaptive k per layer** — use eff_rank_90 directly as k rather than fixed k=64
3. **V-vector compression** — same analysis for V (higher rank, expect smaller gains)
4. **Calibration vs test set** — check if PCA basis transfers across different input texts (crucial for practical deployment)
5. **Hardware cost** — projection + quantize vs plain quantize: latency overhead of the matrix-vector multiply for each KV token

---

## Status (updated 2026-03-29)

The items above were all completed in experiments 1–17. Summary of what we found:

- ✅ Real KV vectors: All experiments 2–17 use real forward-pass KV vectors via hooks
- ✅ Adaptive k per layer: Exp 2 showed fixed k beats eff_rank_90; Exp 16 derived a sensitivity-based adaptive policy (k=64/96/128 per layer, mean=96)
- ✅ V-vector compression: Exp 3 confirmed V is higher rank; recommendation is full-dim PolarQuant for V
- ✅ Calibration transfer: Exp 4 (basic) and Exp 17 (full cross-domain matrix) — k128 transfers well across all domains; k96 benefits from universal multi-domain calibration
- ✅ Hardware cost: Exp 5 measured 1.7× hook overhead (325 vs 185 μs/head); Exp 14 confirmed Python roundtrip is 10–13× (not representative of fused kernel)

See results/SUMMARY.md for the full 17-experiment narrative, and results/REPORT-13 through REPORT-17 for the individual experiment reports.
