# Experiment 8: Layerwise Mixed Compression Policy

## Hypothesis

Compressing early layers causes error cascades that destroy mid/late-layer
attention fidelity (Exp 7 showed early layers at 63% top-1 match, mid/late
collapsing to 26-29%). Protecting early layers from compression should keep
the hidden state clean and reduce perplexity degradation.

**Goal:** Find a mixed compression policy that keeps PPL within 10-20% of
baseline (2.58) while still achieving meaningful compression.

## Setup

- Model: Qwen3-14B-AWQ (40 layers, 8 KV heads, d_head=128)
- 3 evaluation passages (scientific, historical, philosophical) — same as Exp 6
- Sequence length: 512 tokens
- Compression via forward hooks on k_proj/v_proj outputs
- PCA bases from calibration data (results/kvs.npz)
- Compression types:
  - `sub64/4b`: subspace k=64, 4-bit PolarQuant (256 bits/vector)
  - `sub64/2b`: subspace k=64, 2-bit PolarQuant (128 bits/vector)
  - `fd/4b`: full-dim 128, 4-bit PolarQuant (512 bits/vector)
  - FP16 reference: 128 × 16 = 2048 bits/vector

## Policies Tested

| Policy | L0-9 | L10-19 | L20-29 | L30-39 |
|--------|------|--------|--------|--------|
| baseline | none | none | none | none |
| uniform_k64_4bit | K:sub64/4b | K:sub64/4b | K:sub64/4b | K:sub64/4b |
| uniform_kv_optimal | K:sub64/4b V:fd/4b | K:sub64/4b V:fd/4b | K:sub64/4b V:fd/4b | K:sub64/4b V:fd/4b |
| protect_early_10 | none | K:sub64/4b V:fd/4b | K:sub64/4b V:fd/4b | K:sub64/4b V:fd/4b |
| protect_early_20 | none | none | K:sub64/4b V:fd/4b | K:sub64/4b V:fd/4b |
| graduated | none | K:sub64/4b | K:sub64/4b V:fd/4b | K:sub64/2b V:fd/4b |
| k_only_all | K:sub64/4b | K:sub64/4b | K:sub64/4b | K:sub64/4b |

## Results

| Policy | P0 (sci) | P1 (hist) | P2 (phil) | Mean PPL | Rel PPL | CR |
|--------|----------|-----------|-----------|----------|---------|-----|
| baseline | 2.09 | 2.93 | 2.73 | **2.58** | 1.00x | 1.00x |
| protect_early_20 | 2.86 | 6.43 | 4.30 | **4.53** | 1.75x | 1.68x |
| protect_early_10 | 4.43 | 9.83 | 6.77 | **7.01** | 2.71x | 2.56x |
| uniform_k64_4bit | 4.82 | 10.62 | 7.88 | **7.78** | 3.01x | 1.78x |
| k_only_all | 4.82 | 10.62 | 7.88 | **7.78** | 3.01x | 1.78x |
| uniform_kv_optimal | 4.93 | 11.45 | 8.38 | **8.25** | 3.19x | 5.33x |
| graduated | 5.06 | 13.20 | 7.84 | **8.70** | 3.37x | 2.10x |

## Threshold Analysis

**Within 10% of baseline (PPL ≤ 2.84):** none
**Within 20% of baseline (PPL ≤ 3.10):** none
**Within 50% of baseline (PPL ≤ 3.87):** none
**Within 100% of baseline (PPL ≤ 5.16):** protect_early_20 (4.53, +75%)

No policy achieves the 10-20% PPL target. The best policy (protect_early_20)
still degrades PPL by 75%. Subspace + PolarQuant compression at these bit rates
fundamentally distorts the KV cache too much for low-degradation operation.

## PPL vs Compression Ratio (Pareto Frontier)

| Policy | Mean PPL | Rel PPL | CR | Pareto? |
|--------|----------|---------|-----|---------|
| uniform_kv_optimal | 8.25 | 3.19x | 5.33x | **YES** |
| protect_early_10 | 7.01 | 2.71x | 2.56x | **YES** |
| protect_early_20 | 4.53 | 1.75x | 1.68x | **YES** |
| graduated | 8.70 | 3.37x | 2.10x | no |
| uniform_k64_4bit | 7.78 | 3.01x | 1.78x | no |
| k_only_all | 7.78 | 3.01x | 1.78x | no |

Three Pareto-optimal points:
1. **protect_early_20** — best PPL (4.53) at modest compression (1.68x)
2. **protect_early_10** — middle ground (7.01 PPL, 2.56x CR)
3. **uniform_kv_optimal** — highest compression (5.33x) at worst PPL (8.25)

## Does Protecting Early Layers Reduce the Cascade?

**Yes, substantially — but not enough to hit the target.**

| Policy | Layers compressed | Mean PPL | Gap recovered |
|--------|-------------------|----------|---------------|
| uniform_kv_optimal | 40/40 | 8.25 | (reference) |
| protect_early_10 | 30/40 | 7.01 | 21.9% |
| protect_early_20 | 20/40 | 4.53 | **65.7%** |

- Protecting L0-9 recovers 21.9% of the PPL gap (8.25 → 7.01)
- Protecting L0-19 recovers 65.7% of the PPL gap (8.25 → 4.53)
- The relationship is roughly linear: each 10 protected layers recovers ~30% of the gap

This confirms the cascade hypothesis: early-layer compression errors propagate
and amplify through subsequent layers. However, even with 20 layers uncompressed,
the remaining 20 compressed layers still cause 75% PPL degradation.

## Key Observations

1. **K compression dominates the damage.** `k_only_all` (K compressed, V untouched)
   produces identical PPL to `uniform_k64_4bit` (7.78 for both). Adding V compression
   (`uniform_kv_optimal`) only adds +6% PPL (7.78 → 8.25) despite a 3x higher
   compression ratio (1.78x → 5.33x). **V compression is nearly free in PPL terms.**

2. **Graduated policy is counterproductive.** The 2-bit K compression on L30-39
   makes `graduated` (8.70) worse than `uniform_kv_optimal` (8.25), despite
   compressing fewer layers. The 2-bit K subspace quantization is too aggressive.

3. **The cascade is real but compression errors are also local.** Even if early
   layers are pristine, compressing L20-39 alone still causes 1.75x PPL. This
   suggests compression errors are not purely cascaded — each layer's own
   reconstruction error contributes directly to PPL loss.

4. **Passage sensitivity varies.** P1 (historical) consistently shows the worst
   degradation (2-4x worse than P0/P2), suggesting the model is more sensitive
   to KV distortion for certain text domains.

## Recommended Policy

**For maximum compression:** `uniform_kv_optimal` (5.33x CR, 3.19x PPL)
- Best bits-per-PPL efficiency
- V compression adds 3x compression for only +6% PPL over K-only

**For best quality:** `protect_early_20` (1.68x CR, 1.75x PPL)
- Only policy that keeps mean PPL under 5
- But 1.68x compression may not justify the 75% PPL cost

**Bottom line:** Subspace k=64 + PolarQuant at 2-4 bits/dim is too aggressive
for near-lossless KV cache compression on this model. To achieve <20% PPL
degradation, future work should explore:
- Higher bit rates (8-bit quantization)
- Larger subspace dimensions (k=96 or k=112)
- Learned quantization (vs uniform scalar)
- Per-head adaptive k selection based on effective rank
- Token-level selective compression (compress only non-critical positions)
