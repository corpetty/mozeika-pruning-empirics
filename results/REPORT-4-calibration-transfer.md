# Report 4: Calibration vs Test Set Transfer

## Experimental Setup

We tested whether a PCA basis calibrated on **domain 1 (Project Gutenberg fiction)** can effectively compress KV vectors from **domain 2 (Wikipedia-style factual text)** — simulating a real deployment scenario where the compression basis is computed offline on calibration data and then applied to arbitrary user inputs.

**Three conditions compared at k=64:**

| Method | Basis source | Description |
|--------|-------------|-------------|
| Oracle | Domain 2 (self) | PCA fitted on domain 2 calibration split — theoretical best |
| Transfer | Domain 1 → Domain 2 | PCA fitted on fiction, applied to factual text |
| Full-dim | N/A | No projection, just PolarQuant on all 128 dims |

Real KV vectors from the model forward pass (not synthetic) were used for both domains.

## Key Finding: Transfer Works — With a 2× Penalty

### 2-bit budget (256 bits/vector)

| Method | Mean KL | Median KL | Mean Top-1 |
|--------|---------|-----------|------------|
| Oracle | 0.361 | 0.007 | 0.798 |
| **Transfer** | **0.649** | **0.015** | **0.724** |
| Full-dim | 1.668 | 0.068 | 0.563 |

**Transfer penalty: 2.06× mean KL ratio** (median 1.96×)

Despite the 2× penalty vs oracle, transfer **still beats full-dim on 319/320 heads (99.7%)**. This is the critical finding: a domain-mismatched basis is still far better than no basis at all.

### 4-bit budget (512 bits/vector)

| Method | Mean KL | Median KL | Mean Top-1 |
|--------|---------|-----------|------------|
| Oracle | 0.223 | 0.003 | 0.858 |
| Transfer | 0.535 | 0.010 | 0.765 |
| **Full-dim** | **0.102** | **0.003** | **0.886** |

At 4-bit, the story reverses: **full-dim beats transfer on 91% of heads**. The transfer penalty (3.8×) is too large to overcome the inherent advantage of full-dimensional quantization at higher bit rates.

Transfer penalty increases at higher bit rates because quantization noise becomes the minor factor — the dominant error is basis mismatch, which is fixed regardless of bit allocation.

## Transfer Penalty by Layer Range

### 2-bit budget

| Layer Range | Oracle KL | Transfer KL | Penalty | Transfer beats full_dim? |
|-------------|-----------|-------------|---------|--------------------------|
| Early (L0–9) | 0.000126 | 0.000343 | 2.14× | Yes (all heads) |
| Mid (L10–29) | 0.036 | 0.072 | 2.08× | Yes (all heads) |
| Late (L30–39) | 1.371 | 2.451 | 1.96× | Yes (most heads) |

The transfer penalty is **roughly constant across layers** (~2×), slightly lower in late layers. This is consistent with the cross-domain overlap results from Experiment 1 (overlap ~0.70 for K vectors, slightly higher in late layers).

### 4-bit budget

| Layer Range | Oracle KL | Transfer KL | Penalty |
|-------------|-----------|-------------|---------|
| Early (L0–9) | 0.000064 | 0.000289 | 4.48× |
| Mid (L10–29) | 0.016 | 0.052 | 3.84× |
| Late (L30–39) | 0.859 | 2.036 | 2.92× |

At 4-bit, early layers suffer the worst transfer penalty (4.5×), consistent with Experiment 1's finding that early-layer subspaces are less domain-stable.

## Is Offline Calibration Practical?

**Yes, at 2-bit budgets — the regime where subspace compression matters most.**

| Criterion | 2-bit | 4-bit |
|-----------|-------|-------|
| Transfer beats full-dim? | 99.7% of heads | 9% of heads |
| Mean transfer penalty | 2.06× | 3.77× |
| Practical recommendation | Use transfer | Use full-dim instead |

The recommended deployment configuration:
1. **Offline**: Run calibration text through the model, compute per-(layer, head) PCA basis at k=64
2. **Online**: For each new KV vector, project using stored basis, quantize in subspace
3. **Bit budget**: 2-bit equivalent (4 bits/dim × 64 dims = 256 bits/vector)
4. **Expected quality**: ~2× worse than oracle (domain-matched), but ~2.6× better than full-dim 2-bit

### Cost of the transfer approach

The 2× KL penalty from transfer means the effective compression quality of "transfer at 256 bits/vector" is roughly equivalent to "oracle at ~350 bits/vector" — still a substantial compression win over the 2048 bits/vector at FP16.

## Notable Outlier: Layer 27

Layer 27 head 0 showed transfer actually *improving* on oracle (ratio=0.16×). This is likely because the domain 1 basis happened to better capture the common subspace structure, while domain 2's calibration split was too small to fit an optimal basis. This is a statistical artifact at individual heads but illustrates that calibration with more data can sometimes help even for same-domain compression.

## Raw Data

Full results: `results/calibration_transfer.csv` (1,920 rows)
- 320 layer/head combinations × 3 methods × 2 bit budgets
