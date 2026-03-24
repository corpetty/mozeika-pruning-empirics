# Pivot Decision — Mozeika Pruning Framework

*2026-03-24*

---

## What We Set Out to Do

Implement and validate the Mozeika & Pizzoferrato (2026) statistical mechanics pruning framework. The core claims were:

1. **Phase transition:** a critical ρ_c where the Glauber dynamics sharply recover the true sparse mask.
2. **Rényi sharpening:** the replica parameter n controls the sharpness of this transition.
3. **Practical advantage:** this framework should produce better sparse networks than standard methods.

---

## What We Found

### Phase transition theory: linear perceptrons only (CONFIRMED NEGATIVE)

The sharp phase transition is **real and reproducible** in linear perceptrons (Stages 1–2, Exp 16). However:

- **Does NOT generalize to MLPs** (Exp 20): Hamming stays ~0.45 across all ρ. The non-convex loss landscape traps Glauber in locally optimal but globally wrong masks.
- **Does NOT generalize to CNNs** (Exp 22): On MNIST/LeNet-300-100, there is no sharp transition — just gradual accuracy degradation identical to magnitude pruning.
- **ρ_c formula is not predictive** (Exp 24): The Mozeika formula ρ_c = 2√(αη) is off by 100–20,000×. Even our empirical fit only works ~40% of the time.

**Conclusion:** The phase transition is a property of convex (linear) problems, not a general feature of neural network pruning.

### Rényi sharpening: does not exist (CONFIRMED NEGATIVE)

- **Multi-replica at T=0** (Exp 16): n=1,2,4,8 give identical results at zero temperature.
- **Finite temperature** (Exp 23): Sweeping T_h from 1e-6 to 1e-3 never improves over MAP. The "Rényi window" does not exist for these parameters.

**Conclusion:** The replica/Rényi connection is theoretically elegant but empirically vacuous in the MAP regime we can actually compute.

### Rho energy penalty: may produce better high-sparsity masks (NEEDS CONFIRMATION)

- **Exp 21 raw results:** Mozeika MSE=0.334 at 90% sparsity vs Mag+Retrain MSE=0.811 — a 2.4× advantage.
- **BUT:** sparsity control was broken (target 25% → actual 15%; target 50% → actual 59%), making the comparison unfair.
- **Exp 25** re-runs the comparison with iterative ρ adjustment to guarantee ±2% sparsity matching.

---

## Decision Tree

```
Exp 25 result
  │
  ├─ POSITIVE (Mozeika beats Mag+Retrain at ≥2 of {75%, 85%, 90%})
  │     │
  │     ├─ Pivot to "ρ as a pruning objective" framing
  │     │   - The rho energy penalty in the double-well potential acts as
  │     │     an implicit structured sparsity regularizer
  │     │   - Different from L1: penalizes mask entropy, not weight magnitude
  │     │   - Frame as "energy-based pruning" not "phase transition pruning"
  │     │
  │     ├─ GPT-2 experiment (go/no-go criteria below)
  │     │   - Apply Mozeika energy scoring to attention head pruning
  │     │   - Target: 50% head removal with <5% perplexity increase
  │     │   - Baseline: magnitude + Fisher info pruning
  │     │   - If GPT-2 works: write "Energy-Based Pruning" workshop paper
  │     │   - If GPT-2 fails: write method paper on synthetic benchmarks only
  │     │
  │     └─ Timeline: 1 week GPT-2 experiment, 2 weeks paper writing
  │
  └─ NEGATIVE (Mozeika advantage vanishes with proper sparsity matching)
        │
        ├─ The entire advantage was an artifact of unfair comparison
        │
        ├─ Write negative result paper:
        │   "Phase Transitions in Neural Network Pruning:
        │    Theory Confirmed for Linear Models, Does Not Generalize"
        │   - Phase transition confirmed in perceptrons (positive)
        │   - Does not transfer to MLPs, CNNs, or real architectures (negative)
        │   - Rényi sharpening not observed (negative)
        │   - No practical advantage over magnitude pruning (negative)
        │   - Contribution: saves others from pursuing this direction
        │
        └─ Stop further development
```

---

## Go/No-Go Criteria for GPT-2 Experiment

**Prerequisites (all must be true):**
1. Exp 25 shows Mozeika MSE < 0.95 × Mag+Retrain MSE at ≥2 of {75%, 85%, 90%}
2. Sparsity matching within ±2% at those levels (no control artifacts)
3. The advantage is statistically significant (non-overlapping std error bars)

**GPT-2 success criteria:**
- Prune 50% of attention heads with <5% perplexity increase on WikiText-2
- Outperform magnitude-based head importance (Michel et al., 2019) by ≥1% perplexity
- Runtime: energy scoring must complete in <1hr on a single GPU

**GPT-2 failure criteria (stop immediately):**
- Perplexity increase >10% at 50% head pruning
- No improvement over magnitude baseline at any pruning level
- Energy scoring takes >4hr (impractical)

---

## What We Learned

1. **Phase transitions in optimization are fragile.** They depend on convexity, and neural networks are not convex. Don't assume theory proven on linear models transfers.

2. **Sparsity control matters more than the pruning criterion.** Exp 21's "result" was mostly a sparsity matching artifact. Always measure and match actual sparsity.

3. **The rho energy penalty is the interesting part, not the phase transition.** The double-well potential V(h) = α·h²(h-1)² + ρ/2·h is a differentiable approximation to a binary mask that can be used as a pruning objective. This idea is independent of statistical mechanics and may have standalone value.

4. **Negative results have value.** The Rényi/replica connection is widely cited but has never been empirically tested in the pruning context. Showing it doesn't work saves the community time.
