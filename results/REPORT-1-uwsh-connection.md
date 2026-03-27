# Report 1: UWSH Connection — Principal Angle Analysis

## What We Measured and Why It Matters

We tested the **Universal Weight Subspace Hypothesis (UWSH)** in the context of KV cache vectors from Qwen3-14B-AWQ. The UWSH predicts that the principal subspaces learned by different components of a neural network share significant overlap. If true for KV caches, this would enable:

1. **Shared projection bases** across layers/heads (reducing calibration and storage overhead)
2. **Domain-agnostic compression** (a basis fitted on one text domain works on others)
3. **Cross-layer KV sharing** schemes that exploit subspace alignment

We computed **principal angles** between PCA subspaces (retaining 90% variance) for three comparison types:

- **(a) Cross-layer**: Adjacent layers (L_i vs L_{i+1}), all heads pooled per layer
- **(b) Cross-head**: All head pairs within each layer
- **(c) Cross-domain**: Same layer/head, but PCA fitted on fiction (Project Gutenberg) vs. factual text (Wikipedia-style)

Overlap is reported as mean cosine of principal angles: **1.0 = identical subspaces, 0.0 = orthogonal**.

## Key Results

### K Vectors

| Metric | Mean Overlap | Min | Max |
|--------|-------------|-----|-----|
| Cross-layer (adjacent) | **0.5649** | 0.2645 (L0→L1) | 0.7236 (L24→L25) |
| Cross-head (within layer) | **0.4559** | 0.1601 | 0.6724 |
| Cross-domain (fiction vs factual) | **0.7029** | 0.4585 | 0.8566 |

**K vectors by layer range:**

| Layer Range | Cross-Layer | Cross-Head | Cross-Domain |
|-------------|-------------|------------|--------------|
| Early (L0–9) | 0.3811 | 0.3363 | 0.6242 |
| Mid (L10–29) | 0.6186 | 0.4771 | 0.7242 |
| Late (L30–39) | 0.6495 | 0.5332 | 0.7392 |

### V Vectors

| Metric | Mean Overlap | Min | Max |
|--------|-------------|-----|-----|
| Cross-layer (adjacent) | **0.7356** | 0.6274 (L0→L1) | 0.7847 (L23→L24) |
| Cross-head (within layer) | **0.5464** | 0.2453 | 0.7217 |
| Cross-domain (fiction vs factual) | **0.6403** | 0.4339 | 0.7420 |

**V vectors by layer range:**

| Layer Range | Cross-Layer | Cross-Head | Cross-Domain |
|-------------|-------------|------------|--------------|
| Early (L0–9) | 0.6821 | 0.4490 | 0.5744 |
| Mid (L10–29) | 0.7498 | 0.5707 | 0.6635 |
| Late (L30–39) | 0.7634 | 0.5951 | 0.6600 |

## Key Observations

### 1. Cross-domain overlap is highest for K vectors (0.70)
The K-vector principal subspaces are remarkably stable across text domains (fiction vs. factual). With a mean overlap of **0.70**, a PCA basis calibrated on one domain should transfer well to another. This is the most practically important finding — it means **offline calibration is viable**.

### 2. V vectors have stronger cross-layer alignment (0.74) than K vectors (0.56)
Adjacent layers share more V-subspace structure than K-subspace structure. This suggests V vectors live in a more globally consistent subspace, potentially enabling shared V-projection matrices across nearby layers.

### 3. Cross-head overlap is lowest (~0.46 for K, ~0.55 for V)
Different heads within a layer have the most divergent subspaces. This confirms that **per-head bases are necessary** — sharing a single projection across all heads within a layer would lose significant information.

### 4. Early layers are outliers
Layer 0→1 consistently shows the lowest overlap across all metrics. Early layers (L0–9) have substantially lower alignment than mid/late layers. This aligns with the known finding that early transformer layers perform more diverse, input-dependent computations.

### 5. Late layers converge
Mid-to-late layers (L10–39) show consistently higher overlap (0.62–0.76), suggesting these layers have stabilized their representational geometry. Compression with shared bases would work best in this regime.

## Implications for KV Compression

| Finding | Implication |
|---------|------------|
| Cross-domain K overlap = 0.70 | A single offline-calibrated PCA basis per (layer, head) should work across domains with modest degradation |
| Cross-head overlap = 0.46 | Per-head projection bases are required; cannot share across heads |
| Cross-layer K overlap = 0.56 (late: 0.65) | Moderate opportunity for sharing bases across 2–3 adjacent late layers, but not globally |
| V cross-layer overlap = 0.74 | Stronger case for cross-layer V-basis sharing — could reduce basis storage by 2–3× for V |
| Early layer divergence | Layers 0–9 need individual treatment; the "easy" compression targets are layers 10+ |

### Practical savings estimate
- **Current**: Per-(layer, head) basis = 40 layers × 8 heads = 320 bases to store
- **With cross-layer sharing (late layers only)**: Could group into ~15 layer-clusters × 8 heads ≈ 120 bases (~2.7× reduction in basis storage)
- **Cross-domain transfer**: A single calibration run suffices for deployment across text types (no per-domain recalibration needed)

## Raw Data

Full results: `results/uwsh_connection.csv` (2,958 rows)
- 78 cross-layer pairs (39 K + 39 V)
- 2,240 cross-head pairs (1,120 K + 1,120 V)
- 640 cross-domain pairs (320 K + 320 V)
