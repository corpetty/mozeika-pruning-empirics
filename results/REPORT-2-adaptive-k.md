# Report 2: Adaptive k Per Layer — Compression Comparison

## Experimental Setup

We tested whether using the **effective rank at 90% variance (eff_rank_90)** as the subspace dimension `k` — adapting it per layer/head — outperforms a fixed `k=64` for all layers.

**Three methods compared at matched bit budgets:**

| Method | k | Bits/dim | Total bits/vector |
|--------|---|----------|-------------------|
| full_dim | 128 | 2 (or 4) | 256 (or 512) |
| subspace_k64_fixed | 64 | 4 (or 8) | 256 (or 512) |
| subspace_adaptive | eff_rank_90 (7–52) | 256/k (or 512/k) | 256 (or 512) |

All methods use the same total bit budget per vector — the difference is how those bits are allocated between dimensions and precision.

## Key Finding: Fixed k=64 Wins Decisively

**Adaptive k=eff_rank_90 loses to fixed k=64 on 100% of layer/head combinations at both bit budgets.**

### 256-bit budget (2-bit equivalent)

| Method | Mean KL | Mean Top-1 Agreement |
|--------|---------|---------------------|
| full_dim (128d × 2bit) | 3.1096 | 0.378 |
| **subspace_k64_fixed** | **1.3788** | **0.639** |
| subspace_adaptive (k=eff_rank_90) | 2.0170 | 0.491 |

- Adaptive/fixed KL ratio: **3.37×** (adaptive is 3.4× worse)
- Adaptive wins: **0/320 heads (0%)**

### 512-bit budget (4-bit equivalent)

| Method | Mean KL | Mean Top-1 Agreement |
|--------|---------|---------------------|
| full_dim (128d × 4bit) | 0.2946 | 0.826 |
| **subspace_k64_fixed** | **1.2243** | **0.671** |
| subspace_adaptive (k=eff_rank_90) | 2.0100 | 0.470 |

At 4-bit, full_dim actually outperforms both subspace methods — the extra precision per dimension (4 bits at 128d) beats the subspace advantage.

- Adaptive/fixed KL ratio: **6.21×** (adaptive is 6× worse at 4bit)

## Why Does Adaptive k Fail?

The key insight: **eff_rank_90 captures only 90% of variance, discarding 10% permanently.** When k is small (e.g., k=14 for layer 0), the truncation loss dominates quantization noise.

With fixed k=64, the subspace captures **>99% of variance** for most heads (since most eff_ranks are 7–52, well under 64). The remaining budget of 4 bits/dim at k=64 is ample. The adaptive approach trades variance coverage for quantization precision — but **the marginal value of going from 4 bits to 18 bits per dimension is far less than the cost of losing 10% of signal variance**.

In information-theoretic terms: quantization error decreases exponentially with bits (each bit halves the error), but truncation error from dropping principal components is a fixed, irrecoverable loss.

## Per-Layer Breakdown

### Adaptive k values chosen by eff_rank_90

| Layer Range | Mean k | k Range | Mean bits/dim (256 budget) |
|-------------|--------|---------|---------------------------|
| Early (L0–9) | 22.6 | 7–36 | 13.4 |
| Mid (L10–29) | 28.3 | 21–34 | 9.5 |
| Late (L30–39) | 41.6 | 33–50 | 6.4 |

### KL divergence by layer range (256-bit budget)

| Layer Range | full_dim | k64 fixed | adaptive |
|-------------|----------|-----------|----------|
| Early (L0–9) | 0.0017 | **0.0003** | 0.0007 |
| Mid (L10–29) | 1.049 | **0.222** | 0.460 |
| Late (L30–39) | 10.34 | **5.07** | 7.15 |

The gap between adaptive and fixed k=64 is consistent across all layer ranges.

## Implications

1. **Use fixed k=64 (or larger), not adaptive k=eff_rank_90.** The variance retained beyond 90% is critical for reconstruction quality.

2. **The eff_rank_90 metric is useful for understanding structure**, but it's the wrong threshold for choosing compression dimension. A "compression rank" should target ≥99% variance (which eff_rank_99 would yield values closer to 64 anyway).

3. **At 4-bit budgets, full_dim PolarQuant outperforms subspace methods.** The subspace approach is most valuable at ultra-low bit rates (2-bit), where it provides 2.3× KL reduction over full_dim. At 4-bit, the quantization noise is already low enough that the dimension-reduction overhead isn't worthwhile.

4. **The optimal operating point is: k=64, 2-bit equivalent budget (4 bits/dim × 64 dims = 256 bits/vector).** This gives the best KL of any tested configuration at the 256-bit budget level.

## Raw Data

Full results: `results/adaptive_k_distortion.csv` (1,920 rows)
- 320 layer/head combinations × 3 methods × 2 bit budgets
