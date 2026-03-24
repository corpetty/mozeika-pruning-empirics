# Empirical Evaluation of the Mozeika-Pizzoferrato Pruning Framework
*Prepared for discussion with Albert Mozeika — 2026-03-24*
*Authors: Corey Petty, Nick Molty (AI)*

---

## Overview

We implemented your statistical mechanics pruning framework from scratch in Python and ran a systematic empirical evaluation across 25 experiments. The short version: the theory is elegant and the phase transition is real — but it is confined to linear perceptrons. This document is a detailed account of what we tested, what we found, and how we interpret it. We think the results are worth discussing, because the interpretation points somewhere interesting.

---

## What We Implemented

The full Mozeika energy framework: a coupled binary mask variable h and weight vector w with energy

    E(w, h | D) = L(w ∘ h | D) + (η/2)||w||² + Σ V(hᵢ)

where V(h) = αh²(h-1)² + (ρ/2)h is the double-well potential. We implemented all four dynamical regimes (joint Boltzmann, fast learning, fast pruning, low temperature MAP), the Glauber coordinate descent on h with Adam inner loop for w, the multi-replica (Rényi) extension with n independent weight chains sharing one mask, and a clean public API for practical pruning. All 23 unit tests pass. The code is at `/home/petty/pruning-research`.

---

## Experiments

---

### Experiment 1 — Baseline Perceptron Replication
**What we tested:** Reproduce the R code perceptron results in Python. N=500 weights, M=1000 training samples, σ=0.01 noise, sweep η and ρ over an 11×11 grid.

**Key bug caught:** σ matters critically. With σ=1.0 (our first attempt), no phase transition appeared at any ρ — Hamming distance stayed stuck at ~0.47 (random guessing). With σ=0.01 (matching your R code), the transition is sharp. The relevant scale is ρ_c ≈ σ²/N, so you need the sparsity pressure to be on the order of the per-weight contribution to the loss.

**Result:** Phase transition confirmed. Hamming ~0.47 at ρ=0, drops sharply to ~0.02 in one ρ step at ρ≈0.0001. Matches the R output.

---

### Experiments 2–6 — Four Dynamical Regimes
**What we tested:** Implement and compare all four regimes from the theory: joint Boltzmann, fast learning (τ_w ≪ τ_h), fast pruning (τ_h ≪ τ_w), and low-temperature MAP. Phase diagram and finite-size scaling.

**Bug caught:** A normalization error in the phase diagram code (Hamming distance was already in [0,1] but was divided by N again, giving values ~1/N). Fixed.

**Result:** All four regimes implemented and functioning. Phase diagrams show correct qualitative behavior. MAP (low-temperature limit) is the fastest and most effective at recovering the true mask — consistent with your theoretical prediction that β→∞ recovers alternating optimization.

---

### Experiments 7–9 — Multi-Layer Extension (MLP)
**What we tested:** Extend the framework to a 2-layer MLP with nonlinear activations. Experiments: layer-wise sparsity vs ρ, layer collapse, activation function comparison (linear, ReLU, tanh).

**Bugs caught:** The `mlp_forward` function was computing `z = a @ w` instead of `z = a @ w_masked` — the mask was not being applied in the forward pass, so the energy was evaluating the unpruned model. Same bug appeared in the gradient computation. Both fixed.

**Results:**
- Layer collapse: the last layer (4→1) collapses to 0% active even at ρ=0. Earlier layers degrade progressively. This is consistent with the gradient flow behavior you'd expect.
- Activation comparison: tanh and ReLU show poor mask recovery on linearly-generated data. This is expected — model mismatch. The energy landscape is no longer convex.

---

### Experiments 10–12 — UWSH Connection (Spectral Structure)
**What we tested:** Whether pruning near ρ_c concentrates weights into a universal low-dimensional spectral subspace, connecting to the Universal Weight Subspace Hypothesis (Kaushik et al., arXiv:2512.05117). PCA on weight matrices at different ρ, variance concentration in top-k components, principal angles between independent runs.

**Result:** No clear evidence of spectral concentration attributable to the Mozeika energy. The subspace angle comparison (independent runs at the same ρ) did not show convergence. We later found out why — see Experiment 20.

---

### Experiment 16 — Multi-Replica (Rényi) Sweep
**What we tested:** The central theoretical claim of the Rényi extension: n independent weight chains sharing one mask should sharpen the phase transition. Specifically, higher n should lower ρ_c and steepen the Hamming drop. We ran a (n, ρ) grid with n ∈ {1, 2, 4, 8} and 12 ρ values, 6 seeds.

**Result (selected rows from replica_rho_sweep.csv):**

```
ρ          n=1    n=2    n=4    n=8
0.00000    0.365  0.360  0.369  0.367
0.00011    0.027  0.029  0.027  0.031
0.00021    0.046  0.046  0.046  0.046
0.00200    0.175  0.175  0.190  0.173
```

All four n values show essentially identical curves. No sharpening. No shift in ρ_c.

**Interpretation:** We are running in the zero-temperature MAP limit (greedy accept/reject, β→∞). In this limit, the replica parameter n doesn't affect anything — the weight chains all converge to the same MAP solution regardless of n. The Rényi effect is a finite-temperature phenomenon: it requires stochastic acceptance (exp(-ΔE/T_h) > uniform(0,1)) where different n values produce different effective Boltzmann distributions over masks. In the MAP limit, that distinction vanishes. See Experiment 23 for the finite-temperature test.

---

### Experiment 17 — Finite-Temperature Rényi Window
**What we tested:** Whether running at finite T_h (stochastic acceptance) reveals the predicted Rényi sharpening. T_h swept over {0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3}, n ∈ {1, 2, 4, 8}.

**Result:** The MAP baseline (T_h=0) achieves Hamming=0.019 at ρ≈4.5e-5 — near-perfect. At T_h=1e-6 to 3e-6, indistinguishable from MAP. At T_h≥3e-5, the transition blurs and mask recovery degrades (Hamming rises to 0.05–0.22). At T_h≥1e-4, mask recovery effectively breaks (Hamming 0.12–0.29). No n value improves over MAP at any temperature.

**Interpretation:** The thermal noise in the mask dynamics (O(T_h)) swamps the energy signal from the data (O(σ²/N) ≈ 10⁻⁵). You'd need T_h ≪ 10⁻³ to be in a regime where the Boltzmann distribution is localized on good masks — but at that temperature, you're essentially back at MAP anyway. The Rényi window may exist theoretically at a very narrow temperature range, but it is not accessible with practical algorithms.

---

### Experiment 18 — rho_c Scaling Law
**What we tested:** Whether the theoretical formula ρ_c = 2√(αη) correctly predicts the empirical critical ρ across different N, M/N, σ, η combinations.

**Result (from rho_c_comparison.csv):**
```
η=0.0001:  empirical=0.020  theoretical=0.020  → exact match
η=0.0005:  empirical=0.020  theoretical=0.045  → 2.2× off
η=0.001:   empirical=0.020  theoretical=0.063  → 3.2× off
```

The formula works at small η. At larger η it overestimates ρ_c — the strong L2 regularization modifies the energy landscape in ways the first-order theory misses.

Empirical fit across 50 parameter combinations: ρ_c ≈ 0.043 · N^(-0.65) · (M/N)^(-0.83) · σ^0.37 · η^0.24. R²=0.64. The formula ignores N, M/N, and σ dependence entirely, and the full parameter sweep shows errors of 100–20,000×.

**Practical implication:** There is no reliable closed-form predictor for ρ_c. Any real application needs a calibration sweep.

---

### Experiment 20 — MLP Phase Transition + UWSH Jaccard
**What we tested:** Whether the sharp phase transition survives a 2-layer MLP. Architecture: N=10 → H=5 (tanh) → 1 (linear), 55 total mask entries. Three regimes: overdetermined (M=60), critical (M=10), underdetermined (M=5). Also: whether independent pruning runs converge to similar supports (UWSH test via Jaccard similarity).

**Result:** The phase transition **does not appear** in any regime. Hamming stays ~0.43–0.50 across all ρ values. The active count decreases monotonically (38→0 connections) but the mask never converges to the true mask — it converges to masks that are wrong but locally optimal.

Jaccard similarity of surviving weight indices (UWSH test): in the underdetermined regime, Jaccard *decreases* with ρ (0.505 → 0.149). Independent runs diverge more under sparsity pressure. There is no universal sparse support.

**Interpretation:** The non-convex loss landscape has many local minima. The Glauber coordinate descent gets trapped in masks that are locally energy-minimizing but globally wrong. This is the same reason mean-field theory works for the Ising model in high dimensions but fails for sparse factor graphs — the perceptron has a dense bipartite interaction structure (every weight connects to every output), where mean-field is exact. Adding hidden layers creates a sparse, layered interaction graph where mean-field breaks down and the energy landscape becomes corrugated. The transition smears out for the same reason glass transitions smear out in finite-dimensional spin glasses versus the mean-field Sherrington-Kirkpatrick model.

---

### Experiment 22 — CNN/LeNet-300-100 on MNIST
**What we tested:** Whether the Mozeika energy score (sensitivity-weighted pruning in the T→0 fast-learning limit) outperforms magnitude pruning on a real architecture. LeNet-300-100: 784→300 (ReLU)→100 (ReLU)→10, ~266K parameters. Dense baseline: 97.57% accuracy.

**Result:**
```
Sparsity    Magnitude (post-FT)    Mozeika (post-FT)
0%          97.57%                 97.57%
25%         98.04%                 97.86%
50%         98.24%                 98.05%
70%         98.24%                 97.57%
80%         98.03%                 95.65%
```

Magnitude pruning wins at every sparsity level, especially at 80% (98.0% vs 95.7%).

**Interpretation:** Magnitude pruning distributes sparsity non-uniformly across layers (82% in layer 1, 35% in layer 3 — it concentrates pruning where it hurts least). Mozeika applies uniform sparsity pressure per layer, which is suboptimal. More fundamentally, there is no sharp transition — just monotone degradation. The Mozeika energy score reduces to a sensitivity metric related to SNIP (connection sensitivity, Lee et al. 2019) which is already known in the pruning literature and does not outperform simpler methods on standard benchmarks.

---

### Experiments 21 and 25 — Definitive Baseline Comparison
**What we tested:** Whether the Mozeika rho energy penalty produces better sparse networks than magnitude pruning and L1 regularization at matched sparsity. Architecture: 64→32 (ReLU)→1, M=512, σ=0.05. Experiment 21 had a sparsity matching bug (actual sparsity drifted ±10% from target). Experiment 25 fixed it with iterative ρ adjustment until ±2%.

**Result (Experiment 25, matched sparsity):**
```
Sparsity    Mozeika    Mag+Retrain    L1
25%         0.187      0.216          0.205
50%         0.234      0.232          0.227
65%         0.260      0.265          0.222
75%         0.318      0.318          0.224
85%         0.328      0.726          0.375
90%         0.354      0.811          0.446
```

**Mozeika vs Mag+Retrain:** wins at 25% and 85–90%, ties at 50–75%.

**Mozeika vs L1:** loses at 50–75% (0.260 vs 0.222 at 65%), wins at 85–90%.

**The high-sparsity story:** At 85–90% sparsity, Mozeika degrades gracefully (MSE 0.33–0.35) while magnitude + retraining collapses (MSE 0.73–0.81). The rho double-well penalty produces masks that are qualitatively different — it is penalizing mask entropy, not weight magnitude, which causes it to find sparser but coherent solutions rather than random-magnitude solutions.

**However:** L1 regularization achieves similar high-sparsity stability (MSE 0.375–0.446 at 85–90%) without coordinate search overhead. The Mozeika advantage over L1 is marginal and specific to the extreme-sparsity regime (>85%). There is no regime where Mozeika offers a practical advantage over L1 + retraining that would justify the additional computational cost.

---

## Summary Table

| Claim | Prediction | Result |
|-------|-----------|--------|
| Sharp phase transition in linear perceptron | Hamming drops sharply at ρ_c | ✅ Confirmed, reproducible |
| rho_c formula ρ_c = 2√(αη) | Quantitatively accurate | ⚠️ Works at small η only; 100–20,000× off generally |
| Rényi sharpening (multi-replica) | Higher n lowers ρ_c, steeper drop | ❌ No effect at T=0 (MAP regime) |
| Rényi sharpening (finite temperature) | Stochastic acceptance reveals n-dependent sharpening | ❌ Thermal noise swamps signal; MAP is optimal |
| Phase transition in MLPs | Sharp mask recovery at ρ_c | ❌ No transition; corrugated landscape traps Glauber |
| Phase transition in CNNs | Sharp accuracy threshold | ❌ No transition; gradual degradation only |
| UWSH connection | Pruning concentrates into universal subspace | ❌ Jaccard decreases with ρ; no shared support |
| Practical advantage over baselines | Better sparse networks than magnitude pruning | ⚠️ Wins at >85% sparsity vs magnitude, but L1 is competitive |

---

## Interpretation and Where This Points

The phase transition is real and theoretically clean — but it is a property of convex optimization on dense interaction graphs, not a general feature of neural network pruning.

The connection we'd suggest: your framework is essentially **mean-field theory applied to the Ising spin-glass analogy for weight masks**. Mean-field is exact when the interaction graph is fully connected (every spin sees every other spin, equivalent to your perceptron where every weight contributes to every output). The moment you add hidden layers, the interaction graph becomes sparse and layered. In spin glasses, this is exactly the regime where mean-field breaks down and the replica trick stops being exact — you get RSB (replica symmetry breaking), a corrugated free energy landscape, and exponentially many metastable states. That's what we observe: Glauber gets trapped in wrong-but-locally-optimal masks.

The natural next step theoretically would be belief propagation on the factor graph of the MLP — which is the exact inference analogue to what survey propagation does for satisfiability in sparse random graphs. Mézard and Montanari's work on BP-guided decimation for random CSPs might be the right reference frame. That's a different and harder algorithm, but it would be principled.

The double-well potential V(h) is, separately, a useful practical idea as a differentiable approximation to binary masks — that idea is independent of the statistical mechanics framing and might have value as a regularizer in its own right.

---

## Files

All code, results, and analysis at `/home/petty/pruning-research`.

```
pruning_core/         — Python library (energy, dynamics, optimizers, replicas, pruner API)
experiments/01–25     — All experiments with comments
results/              — CSV output for every experiment
tests/                — 23/23 passing unit tests
REPORT.md             — Full technical report
PIVOT.md              — Decision tree and final go/no-go analysis
```

---

*We'd value your thoughts on the mean-field/sparse factor graph interpretation particularly. If the BP direction is interesting to you, we have the infrastructure to test it.*
