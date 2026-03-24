# Pruning Research: Master Plan
*Last updated: 2026-03-23*

## What We Know (Validated)

- Mozeika phase transition is real: Hamming distance drops sharply at rho_c in perceptron
- Zero-temperature MAP (Glauber coordinate search + Adam inner loop) works best
- rho_c scales with N and M/N — the published formula 2√(αη) is wrong outside perturbative regime
- Empirical fit: rho_c ~ 0.043 × N^(-0.65) × (M/N)^(-0.83) × σ^0.37 × η^0.24 (R²=0.64)
- Finite-temperature Rényi sharpening not observed at T_h=0.001–0.1 (too hot)
- Perceptron UWSH test is degenerate (convex loss → unique w*)

## What We Don't Know (Open Questions, Priority Order)

1. **Does the phase transition survive in non-linear networks?** (MLP, CNN, transformer)
2. **Does Mozeika pruning beat simpler baselines at matched sparsity?** (magnitude, L1, random)
3. **Does rho_c formula generalize — can we predict it before training?**
4. **Is there a real UWSH connection? Do independent runs prune same weights in non-convex regime?**
5. **Can we apply this layer-by-layer to a real LLM and maintain perplexity?**
6. **What is the minimal T_h window where Rényi sharpening appears?** (need T_h ≪ 10^-3)

## Experimental Roadmap

### Phase 1: Theory Validation Beyond Perceptron (tonight)

**Exp 20: MLP Phase Transition + UWSH**
- 2-layer MLP [N→H→1] in underdetermined regime (M < N)
- Measure: does sharp phase transition in Hamming persist?
- Measure: Jaccard similarity of pruned support (h_final) across independent runs
- Key: non-convex loss → multiple solutions → non-trivial support overlap

**Exp 21: Baseline Comparison**  
- At matched sparsity, compare Mozeika-guided pruning vs:
  - Magnitude pruning (remove smallest |w|)
  - L1 regularization (λ||w||_1 penalty, threshold at end)
  - Random pruning (control)
- Metric: test MSE at sparsity levels 0%, 25%, 50%, 75%, 90%
- Network: 2-layer MLP, overdetermined (M=4N), standard supervised task
- Key question: does the phase-transition-guided rho actually find better masks?

**Exp 22: CNN on MNIST**
- LeNet-5 or small CNN (2 conv + 2 FC layers)
- Apply GlauberPruner layer-by-layer (independent rho per layer)
- Measure: test accuracy vs total sparsity curve
- Compare to magnitude pruning baseline at same sparsity
- Key: first real architecture test. If it works here, it generalizes.

**Exp 23: Low T_h Rényi Window**
- Sweep T_h in [1e-6, 1e-5, 1e-4, 5e-4] with n=[1,2,4,8]
- Fix everything else at Exp 16 optimal params (N=60, sigma=0.01, etc.)
- Looking for the narrow window where stochastic acceptance improves over MAP
- If found: characterize the improvement (rho_c shift, transition sharpness)
- If not found at these temps: declare MAP optimal in practical regime

### Phase 2: Real Architectures (this week)

**Exp 25: Small Transformer (GPT-2 small or equivalent)**
- Apply GlauberPruner to attention weight matrices (Q, K, V, O) and MLP layers
- Layer-by-layer rho sweep; find per-layer rho_c
- Measure perplexity on validation set vs total sparsity
- Compare to SparseGPT and magnitude pruning at same sparsity

**Exp 26: Layer-wise rho_c Prediction**
- Use empirical rho_c formula to predict rho_c per layer without running sweep
- If prediction is accurate, we can prune in one shot without calibration

**Exp 27: Structured Sparsity**
- Instead of unstructured (individual weights), test block-structured masks
- 4x4 blocks or row/column structured masks (hardware-friendly)
- Does phase transition survive? Does rho_c shift?

### Phase 3: LLM Pilot (next week)

**Target: LLaMA-7B or Qwen-7B (we have 48GB VRAM)**

**Approach:**
1. Load model in bfloat16 (~14GB)
2. For each linear layer: run Glauber coordinate search with small calibration set (128 samples)
3. Apply mask; measure perplexity on WikiText-103
4. Compare to: dense baseline, SparseGPT (50% sparsity), Wanda

**Success criteria:**
- At 50% sparsity: perplexity increase < 10% over dense → framework is viable
- At 50% sparsity: perplexity better than magnitude pruning → framework adds value
- At 50% sparsity: within 2 PPL of SparseGPT → competitive with SotA

**Failure criteria (hard evidence of dead end):**
- Phase transition disappears in overparameterized regime
- rho_c → 0 for all layers (everything is noise, nothing is prunable)
- Performance drop >20% at 50% sparsity even with optimal rho

### Phase 4: SotA Comparison & Publication Path

**Baselines to beat:**
- SparseGPT (Frantar & Alistarh, 2023) — second-order pruning, 50-60% with minimal degradation
- Wanda (Sun et al., 2023) — magnitude × activation, no weight update needed
- RIA (Zhang et al., 2023) — relative importance, unstructured
- SliceGPT — structured, removes entire rows/columns

**Our potential edge:**
- Principled sparsity pressure via rho (not a heuristic)
- Phase transition gives natural stopping criterion (don't prune past rho_c)
- Replica interpretation suggests ensemble diversity → more robust sparse solutions
- Layer-specific rho_c prediction could reduce calibration cost

**Publication target if results are good:**
- NeurIPS 2026 (submission deadline ~May)
- "Statistical mechanics of LLM pruning: phase transitions and the rho_c scaling law"

## Hardware

- 2× RTX 3090 (24GB each), 64-core EPYC, 503GB RAM
- LLaMA-7B fits in one GPU. 13B needs both. 70B needs CPU offload.
- Run multiple experiment agents in parallel — machine is underutilized overnight

## Success / Failure Criteria Summary

| Question | Hard Failure | Success |
|----------|-------------|---------|
| MLP phase transition | No transition for any MLP | Sharp transition at predicted rho_c |
| Baseline comparison | Magnitude beats Mozeika consistently | Mozeika ≥ magnitude at ≥3 sparsity levels |
| CNN test | Accuracy collapse at any sparsity | Accuracy within 1% of dense at 50% sparsity |
| LLM pilot | PPL > 2× dense at 50% sparsity | PPL within 10% of dense at 50% sparsity |
| UWSH in MLP | Jaccard < 0.2 (random overlap) | Jaccard > 0.5 at rho_c across independent runs |

Any single hard failure at Phase 2+ = pause and re-evaluate. Two failures = dead end, write it up as a negative result.
