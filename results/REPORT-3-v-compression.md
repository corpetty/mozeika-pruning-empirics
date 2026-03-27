# Report 3: V-Vector Compression Analysis

## Experimental Setup

We replicated the compression distortion analysis from the original K-vector study, but applied to **V (value) vectors**. Synthetic V vectors were generated from the measured V singular value spectra (`sv` from `analysis.npz`), then compressed using full-dim PolarQuant and subspace PolarQuant at k={16, 32, 48, 64}, with matched bit budgets at 2, 4, and 8 bits/dim.

## Key Finding: V Vectors Are Harder to Compress via Subspace Projection

### V compression at 2-bit budget (256 bits/vector)

| Method | V Mean KL | K Mean KL | Ratio (V/K) |
|--------|-----------|-----------|-------------|
| full_dim (128d × 2bit) | 2.562 | 3.110 | 0.82× (V better) |
| subspace_k16 | 3.285 | 3.116 | 1.05× |
| subspace_k32 | 2.711 | 2.191 | 1.24× |
| subspace_k48 | 2.266 | 1.657 | 1.37× |
| **subspace_k64** | **1.944** | **1.379** | **1.41×** |

**Two important asymmetries emerge:**

1. **Full-dim: V compresses better than K** (V_KL=2.56 vs K_KL=3.11). V vectors are smoother and more evenly distributed, making uniform quantization more effective.

2. **Subspace: K compresses better than V** (at k=64: K_KL=1.38 vs V_KL=1.94). K vectors are more low-rank, so projecting to 64 dimensions captures more of K's variance than V's.

## Why the Asymmetry?

The root cause is the **effective rank difference**:

| Statistic | K eff_rank_90 | V eff_rank_90 |
|-----------|--------------|--------------|
| Min | 7 | 13 |
| Max | 52 | 79 |
| Mean | **29.6** | **54.2** |

V vectors have nearly **2× the effective rank** of K vectors. This means:
- V vectors spread their variance across more dimensions
- At k=64, subspace projection captures >99% of K variance but only ~90% of V variance
- The truncation loss is much larger for V

## Does the Crossover Point Differ?

**K vectors**: Subspace k=64 beats full-dim at 2-bit (1.38 vs 3.11, 2.3× better) and is close at 4-bit.

**V vectors**: Subspace k=64 beats full-dim at 2-bit (1.94 vs 2.56, 1.3× better) but the margin is much smaller. At 4-bit:

| Method | V 4-bit KL |
|--------|-----------|
| full_dim | **0.241** |
| subspace_k64 | 1.944 |

At 4-bit, **full_dim crushes subspace for V vectors** — the subspace KL at 4-bit (1.94) is barely different from 2-bit (1.94), suggesting the V subspace truncation loss is the binding constraint, not quantization noise.

**Crossover summary:**
- **K**: Subspace wins at ≤2 bits/dim, full_dim wins at ≥4 bits/dim
- **V**: Subspace wins only at 2 bits/dim, and with a smaller margin; full_dim wins at ≥4 bits/dim decisively

## Per-Layer Breakdown (2-bit, k=64)

| Layer Range | V KL (k=64) | V Top-1 | V eff_rank_90 |
|-------------|-------------|---------|---------------|
| Early (L0–9) | 0.0002 | 0.583 | 39.0 |
| Mid (L10–29) | 0.274 | 0.475 | 57.8 |
| Late (L30–39) | 7.229 | 0.440 | 62.3 |

Late layers are the hardest for V compression, consistent with higher effective ranks. Layers 30–39 have V eff_rank ~62, approaching the projection dimension k=64, which leaves almost no room for the subspace to help.

## Practical Implications

### 1. Use different k for K vs V
- **K vectors**: k=64 is ideal (captures >>99% variance for most heads)
- **V vectors**: k=64 captures only ~90% variance on average. Consider k=80–96 for V, or accept higher distortion.

### 2. Or use different strategies entirely
Given V's higher rank, a hybrid approach may be optimal:
- **K**: Subspace PolarQuant at k=64, 4 bits/dim (256 bits/vector) — strong compression
- **V**: Full-dim PolarQuant at 4 bits/dim (512 bits/vector) — simpler, better quality

This gives K at 256 bits/vector and V at 512 bits/vector = **768 bits per (K,V) pair** vs. the 2×128×16 = 4096 bits at FP16, an overall **5.3× compression ratio** while maintaining good quality.

### 3. The subspace advantage is K-specific
The strong subspace compression wins reported in earlier experiments apply primarily to K vectors. V vectors benefit less from subspace projection because their variance is spread across more dimensions. Claims about subspace KV compression should distinguish between K and V.

### 4. V eff_rank increases with layer depth
Unlike K (which has moderate eff_rank throughout), V eff_rank grows steadily from ~39 in early layers to ~62 in late layers. This suggests V representations become increasingly diverse/distributed in deeper layers.

## Raw Data

Full results: `results/v_compression_distortion.csv` (4,800 rows)
- 320 layer/head combinations × 5 methods × 3 bit budgets
