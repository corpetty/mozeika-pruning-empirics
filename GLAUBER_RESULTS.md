# Glauber Dynamics Pruning — Experimental Results

## Setup

LeNet-300-100 MLP on MNIST (fc1: 784→300, fc2: 300→100, fc3: 100→10, ReLU).  
Script: `lenet300_pruning_finite_temp.py` (Alexander M., contributed 2026-03-27).  
Environment: `/home/petty/torch-env` (Python 3.12, PyTorch), CUDA device 1 (RTX 3090).

Architecture uses `MaskedLinear` layers with binary weight masks updated via finite-temperature
Glauber dynamics. Fisher information computes Δ for prune/regrow decisions.
Glauber flip probability = σ(−β_h · Δ).

---

## Experiment Series

### Run 1 — Fixed T_h=1e-7, max_flip_fraction=0.2 (default)
- **Result:** 63.1% sparsity, 97.89% accuracy after 20 rounds
- Flip balance converged to ~0 by round 11 (equilibrium)
- No dead neurons

### Run 2 — Fixed T_h=1e-7, max_flip_fraction=0.5
- **Result:** Same ~63% equilibrium sparsity, ~97.75% accuracy
- Converged by round 3 (faster, same fixed point)
- Confirms max_flip_fraction controls speed, not equilibrium

### Run 3 — T_h=0 (zero temperature / greedy)
- **Result:** 99% sparsity in 5 rounds, accuracy collapsed to 93.55%
- 106 dead neurons in fc1, 20 in fc2
- No regrowth; monotonic pruning

### Run 4 — Temperature anneal T_h=1e-7 → 0, linear over 20 rounds, target=95%
- **Result:** 95.3% sparsity, 97.47% accuracy, reached in round 17
- 21 dead neurons at end; zero dead neurons through round 15 (86% sparsity)
- Stopped early at target

### Run 5 — Anneal 1e-7→0, 20 rounds, target=97%
- **Result:** 98.1% sparsity, 97.32% accuracy (round 18)
- 38 dead fc2 neurons, 0 dead fc1

### Run 6 — Anneal 1e-7→0, 20 rounds, target=99%
- **Result:** Stalled at 98.94% after 20 rounds
- Round 19: 169 dead fc1 neurons appear suddenly

### Run 7 — Anneal 1e-7→0, 20 rounds, target=99% + energy tracking
- Energy decomposition per round (CE + L2 + ρ penalty)
- Rounds 1–15: pure restructuring, zero neuron deaths, weights pruned ~29–53k/round, regrown ~19–24k/round
- Round 19: 148 neurons die at once (fc1 collapse); no rebirths after round 17
- Final: 98.94% sparsity, 97.37% accuracy

**Comparison table (1x schedule):**

| Method                     | Sparsity | Accuracy | Dead neurons |
|----------------------------|----------|----------|--------------|
| Dense baseline             | 0%       | 96.87%   | —            |
| Magnitude pruning 95%      | 95.2%    | 97.26%   | —            |
| Glauber T=1e-7 fixed       | 63.1%    | 97.89%   | 0            |
| Glauber T=0 greedy         | 99.0%    | 93.55%   | 126          |
| Glauber anneal 95% target  | 95.3%    | 97.47%   | 21           |
| Glauber anneal 97% target  | 98.1%    | 97.32%   | 38           |
| Glauber anneal 99% (stall) | 98.94%   | 97.37%   | 186          |

---

## 2 Rounds Per Temperature Experiment

**Motivation:** In equilibrium statistical mechanics ⟨E⟩ is non-increasing with decreasing T.
The 1x schedule showed non-monotone energy steps; 2x was tested to check if extra relaxation
brings the system closer to the quasi-static limit.

**Setup:** Same 20-step linear anneal, 2 Glauber sweeps + weight retraining per step = 40 rounds max. target=99%.

**Result:** Target reached at round 37 (T=5.26e-9). **99.05% sparsity, 97.46% accuracy.**

### Phase structure:

| Phase            | Rounds | Behaviour                                                         |
|------------------|--------|-------------------------------------------------------------------|
| Rapid pruning    | 1–2    | 0% → 35% sparsity; large energy drop                             |
| Restructuring    | 3–30   | 35% → 87% sparsity; ~0 neuron deaths; active prune/regrow cycles |
| Collapse         | 31–37  | Deaths begin (4→6→28→12→22→1→183); target hit at round 37       |

### Energy behaviour:
- Smoother than 1x — second round per temperature recovers energy spikes from the first
- Still not monotone (16 non-monotone steps out of 37); violations smaller than 1x
- True monotonicity would require many more sweeps per temperature step

### vs 1x:
- 1x stalled at 98.94% (never reached 99%); 2x clears the target
- The fundamental dynamics (plateau → cliff → collapse) are unchanged
- The cliff at the final step is still present, just pushed one temperature step later

---

## Three-Way Comparison: Glauber vs Magnitude vs Magnitude+Rewind

**Setup:** All three methods follow the identical 37-round sparsity schedule derived from the
2x Glauber trajectory. Ensures comparison is at matched sparsity at every round.

### Magnitude pruning (no rewind)
- Standard iterative magnitude pruning + finetune after each step
- No regrowth; once pruned, always pruned

### Magnitude + rewind (lottery ticket style)
- After each prune step, surviving weights rewind to pretrained values
- Network retrained from pretrained initialisation under the new mask each round

### Final results at ~99% sparsity (round 37):

| Method                    | Sparsity | Accuracy | Test loss |
|---------------------------|----------|----------|-----------|
| Glauber anneal 2x         | 99.05%   | **97.46%** | **0.064** |
| Magnitude (no rewind)     | 98.95%   | 96.38%   | 0.109     |
| Magnitude + rewind        | 98.95%   | 96.17%   | 0.124     |

### Key findings:

1. **Glauber wins at 99% sparsity** by +1.08% over plain magnitude, +1.29% over magnitude+rewind.
   Loss gap is ~2×.

2. **Rewinding hurts at extreme sparsity.** Pretrained dense weights are a poor initialisation
   for a 99%-sparse network. The lottery ticket rewind benefit appears when rewinding to early
   training checkpoints (Frankle & Carlin 2019 use ~0.1% of training), not dense-pretrained state.

3. **Glauber's advantage is the regrowth mechanism, not the criterion.** The mask stays plastic
   across all 37 rounds — weights flow in and out continuously. By the final round, surviving
   connections are already well-arranged. Magnitude pruning makes a large irreversible jump at
   round 37 with no recovery path.

4. **Rounds 1–34 (0–95% sparsity):** All three methods track within ~0.3% accuracy of each other.
   The regrowth advantage only manifests significantly at extreme sparsity.

5. **Practical implication:** For applications requiring >95% sparsity, continuous mask dynamics
   (Glauber, DST, RigL-style) substantially outperform hard pruning methods regardless of criterion.

---

## Plots

- `energy_vs_rounds.png` — energy components vs round (Run 7, 1x schedule)
- `energy_vs_temperature.png` — energy vs T_h log scale (Run 7)
- `energy_1x_vs_2x.png` — energy trajectory: 1x vs 2x rounds/temp
- `metrics_vs_rounds.png` — accuracy, loss, sparsity vs round: 1x vs 2x
- `acc_vs_loss.png` — accuracy vs loss trajectory coloured by sparsity: 1x vs 2x
- `glauber_vs_magnitude.png` — Glauber 2x vs magnitude pruning (matched schedule)
- `three_way_comparison.png` — Glauber vs magnitude vs magnitude+rewind

---

## Connection to Mozeika & Pizzoferrato (2026)

These experiments sit outside the regime where Mozeika's statistical mechanics formalism applies:

- The system is never at thermal equilibrium (one or two sweeps per temperature step)
- The Hamiltonian co-evolves with the mask (weights retrained between sweeps)
- Mean-field breaks down for the layered MLP factor graph (as established in prior experiments)

The Glauber dynamics here are a heuristic borrowing the *form* of finite-temperature sampling
without the theoretical guarantees. The empirical results are strong, but they do not validate
the Mozeika framework — they demonstrate that the sampling idea is practically useful even when
the theory doesn't apply cleanly.
