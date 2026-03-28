# Research Plan: Statistical Mechanics of Neural Network Pruning
## A Framework-First Approach

---

## What This Is Actually About

The Mozeika & Pizzoferrato paper does something most ML pruning literature doesn't: it derives pruning from first principles using statistical mechanics. The mask vector h isn't a heuristic — it's a thermodynamic variable with its own temperature, timescale, and energy landscape. The phase transition isn't a metaphor — it's a prediction from the replica method that should be numerically verifiable.

This plan is organized around **scientific questions**, not implementation milestones. Code follows questions, not the other way around.

The adjacent work that makes this more interesting than it looks in isolation:

- **UWSH (Kaushik et al. 2025):** 1100+ models converge to shared low-dimensional spectral subspaces regardless of task or initialization. LoRA, pruning, model merging all operate on the same object.
- **Daemon (Jarrad Hope):** Vertically integrated decentralized AI platform — local inference, cloud routing, uncensored serving, heterogeneous training network. Needs principled sparsity for communication efficiency.
- **Rényi connection (unwritten):** The replica parameter n = β_h/β_w is the Rényi entropy order parameter. n=1 is standard Bayesian inference; n→∞ is minimax mask selection. This connection does not appear in the pruning literature.

Your background — computational physical chemistry, statistical mechanics, security — positions you to push this further than most ML people can.

---

## The Central Questions

### Q1: Is the phase transition real in multi-layer networks?
The R perceptron result shows it clearly. Does it survive depth? Does it survive non-linearity? Does it survive overparameterization (width >> necessary)? The replica calculation assumes the thermodynamic limit — what happens at finite, realistic N?

### Q2: Is ρ_c predictable without search?
The replica method gives an analytic expression for the critical pruning pressure ρ_c as a function of (N, M, η, architecture). If that prediction is accurate, you can set ρ directly from data statistics — no grid search, no cross-validation. This would be a practical contribution with immediate impact.

### Q3: Does Mozeika pruning carve out the UWSH subspace?
If all trained networks converge to the same principal directions (UWSH), and pruning near ρ_c concentrates weights into fewer singular directions, then pruning and the universal subspace might be the same object viewed differently. Test: do pruned weight matrices from different seeds have higher subspace overlap than unpruned ones? Does pruning *cause* UWSH alignment?

### Q4: What does the Rényi/replica connection mean practically?
n = β_h/β_w is not just a mathematical curiosity. It parameterizes a family of pruning algorithms:
- n<1: entropy-maximizing, prefers denser masks, more uncertain
- n=1: Bayesian optimal, trades off mask complexity against fit
- n>1: aggressive, minimax, worst-case mask recovery
- n→∞: MAP, greedy coordinate descent (current R implementation)

Are there tasks where n≠∞ is actually better? Federated/heterogeneous settings where you want uncertainty over the mask?

### Q5: Does any of this matter for decentralized training?
Daemon's architecture involves training over heterogeneous consumer hardware with communication constraints. Pruning reduces communication cost. But UWSH suggests all nodes converge to the same subspace anyway — meaning you might only need to communicate the subspace coordinates, not the full weights. Does Mozeika pruning + UWSH give you a principled compression scheme for federated weight sharing?

---

## Research Arc

### Stage 1 — Establish the Baseline (Weeks 1–2)
**Goal:** Clean, reproducible Python implementation that matches the R results exactly.

Not just a port — a library with a clear interface:
```python
energy = MozeikaEnergy(eta=0.001, rho=0.0007, alpha=1.0)
result = GlauberDynamics(energy, regime='fast_learning').run(model, data)
```

Experiments to reproduce:
- Perceptron: 11×11 (η, ρ) grid, Hamming distance curves, phase boundary
- NN exhaustive: 4→3→1, 2^N enumeration, heatmap
- NN Glauber vs exhaustive: approximation gap as function of N

Validation criterion: phase transition location matches R output within 10% across seeds.

The exhaustive enumeration result is theoretically important — it gives you the ground truth energy landscape minimum, not just what Glauber finds. Keep it as a calibration tool even as N grows.

### Stage 2 — Theory Validation (Weeks 3–5)
**Goal:** Test the four theoretical regimes and the replica predictions numerically.

**Four regimes (explicit experiments, not just descriptions):**

1. *Equal timescales* — joint Langevin on w and h simultaneously. Show this interpolates between the fast-learning and fast-pruning limits as τ_w/τ_h varies.
2. *Fast learning* (τ_w ≪ τ_h) — current R implementation. Adam inner loop on w, Glauber outer loop on h. Validate: how many Adam steps approximate "fully equilibrated"? (Convergence test vs K.)
3. *Fast pruning* (τ_h ≪ τ_w) — masks flip fast, weights lag. Expected: finds sparse solutions but weight estimates are noisy. Useful? Probably not, but tests the theory.
4. *Low temperature* (β→∞) — MAP limit, greedy descent. This is the R code. Establish that it's the limit of the stochastic versions.

**Replica prediction validation:**

The replica method predicts ρ_c(η, N, M/N). Numerically:
- Fit sigmoid to empirical Hamming(ρ) curves → extract ρ_c
- Compare to analytic prediction across (η, M/N) grid
- Measure: how accurate is the prediction? Where does it break down?

**Finite-size scaling:**
- N ∈ {50, 100, 200, 500, 1000} at fixed M/N = 2
- Phase transition sharpens as N→∞ (thermodynamic limit prediction)
- Measure the scaling exponent — this is publishable as a standalone result

### Stage 3 — Deep Networks (Weeks 5–7)
**Goal:** Establish whether the theory generalizes beyond the perceptron.

The interesting architectural questions:

*Layer-wise vs joint pruning:* Does pruning each layer independently with its own (η_l, ρ_l) work better than global pruning? The paper suggests layer-specific temperatures prevent layer collapse. Test this explicitly.

*Layer collapse:* Which layers collapse first under uniform ρ? Is it always the first layer? Does width ratio between adjacent layers predict collapse order? This has direct practical relevance for architecture design.

*Depth scaling:* Fix total parameter count N, vary depth. Does ρ_c change? Does the phase transition survive? Deep networks with skip connections are the interesting case — the mask energy couples across layers through the skip path.

*Non-linearity:* The R code has tanh commented out. Implement and compare. GELU is what matters for transformers. The gradient changes but the energy structure should be similar.

The N=10 exhaustive NN result is your calibration anchor here. For small networks you can always check Glauber against the exact solution.

### Stage 4 — UWSH Connection (Weeks 7–9)
**Goal:** Test whether Mozeika pruning and the universal weight subspace are the same object.

This is the most potentially novel contribution.

**Experiment 1 — Spectral structure under pruning:**
- Train identical architectures from different seeds
- Prune with Mozeika at various ρ (including near ρ_c)
- Compute SVD of weight matrices at each ρ
- Measure: effective rank (number of significant singular values), principal angle between subspaces across seeds
- Hypothesis: near ρ_c, subspaces from different seeds converge (pruning *causes* UWSH alignment)

**Experiment 2 — What does pruning remove?**
- Decompose weight matrix W = Σ σᵢ uᵢvᵢᵀ
- After pruning, what fraction of signal is in top-k singular directions?
- Does Glauber preferentially kill weights in low-singular-value directions? (It should if the energy landscape aligns with spectral structure)

**Experiment 3 — LoRA in the pruned subspace:**
- Take a pruned network, apply LoRA fine-tuning on a new task
- Measure: do LoRA adapters align with the surviving weight directions?
- If yes: pruning carves the UWSH subspace explicitly, and LoRA fine-tunes within it

**The theoretical synthesis to write:**
Mozeika's Langevin energy E(w,h|D) has a minimum where w∘h lives in the signal subspace of XᵀX. UWSH says this subspace is architecture-defined, not data-defined. If both are true simultaneously, then the ground truth mask h₀ is implicitly defined by the spectral geometry of the architecture — not by the data. This would explain why LoRA, pruning, and model merging all work: they're all projecting onto the same geometry.

### Stage 5 — Rényi/Replica Experiments (Weeks 9–11)
**Goal:** Test whether the replica parameter n is a useful practical knob.

Implement non-integer n by running n independent w-chains with a shared h:
- Each replica sees the same mask h but optimizes w independently
- The h update uses the ensemble energy (average over replicas weighted by β_h/β_w)
- n=1: single replica (current implementation)
- n=2: two replicas, more conservative mask selection
- n→∞: MAP limit

**Experiments:**
- Sweep n ∈ {0.5, 1, 2, 5, ∞} at the theoretically optimal (η, ρ)
- Measure: Hamming distance, test error, convergence speed
- Does n=1 give genuinely better generalization than n→∞ on held-out data?
- Is there a task-dependent optimal n?

**The federated learning angle:**
In distributed training, different nodes see different data. The effective n for the ensemble of nodes is related to the KL divergence between their data distributions. Does the optimal pruning strategy depend on data heterogeneity? This connects directly to Daemon's training architecture.

### Stage 6 — Practical Pruner (Weeks 11–13)
**Goal:** A PyTorch-compatible implementation that competes with SOTA on real benchmarks.

This stage is conditional on Stage 2 delivering a good ρ_c prediction. If you can predict ρ_c analytically, the pruner doesn't need a grid search — you set (η, ρ) from architecture statistics and run Glauber once.

**Interface:**
```python
pruner = MozeikaPruner(
    eta='auto',          # derived from weight norm statistics
    rho='auto',          # derived from replica prediction for this (N, M)
    regime='fast_learning',
    per_layer=True,
    n_replicas=1         # Rényi parameter
)
model = pruner.fit(model, dataloader)
```

**Benchmarks:**
- MNIST/CIFAR-10 MLP: compare sparsity vs accuracy against magnitude pruning, IMP
- Small transformer (GPT-2 scale): compare against SparseGPT, Wanda
- The honest metric: performance at a given sparsity fraction, not just best-case numbers

**The Daemon angle:**
If the pruner works at the node level in a federated setting — each node prunes locally, shares only the surviving weight subspace — you get communication efficiency from the physics. This is worth a separate write-up for the IFT/Daemon audience.

---

## What This Is Not

- Not a hyperparameter search paper ("we tuned η and ρ and got +0.3%")
- Not a benchmark paper ("our method beats 7 baselines")
- A theory-first paper that validates physical predictions and discovers new structure

The publishable contributions in rough priority:
1. Numerical validation of the phase transition in deep networks (Stage 3)
2. Rényi/replica parameter as a practical pruning knob (Stage 5) — likely novel
3. UWSH + Mozeika = same object (Stage 4) — potentially high impact
4. Predictable ρ_c from architecture statistics (Stage 2) — practical value

---

## Hardware / Infrastructure

- **Stages 1–4:** CPU, runs on laptop. The R perceptron experiment fits in <5 min on a single core.
- **Stage 4 (UWSH at scale):** GPU 1 on bugger (GPU 0 holds Ollama)
- **Stage 5–6 (transformer experiments):** both GPUs, Ollama paused

**Stack:** Python 3.11, PyTorch 2.x, NumPy, SciPy (for replica calculations), matplotlib/seaborn for publication plots. No JAX dependency unless the Langevin dynamics gets complex enough to need it.

---

## Open Questions That Might Redirect Everything

1. If ρ_c is analytically predictable, does exhaustive/Glauber search become unnecessary? (The answer changes the practical contribution entirely.)
2. Does the phase transition survive overparameterization? Modern networks are massively overparameterized — the thermodynamic limit analysis may not apply in the regime N >> M.
3. Is the UWSH subspace the same as the Mozeika ground-truth mask support, or just correlated? The causal direction matters for the theory.
4. What does n < 1 (Rényi order < 1) mean physically — entropy-maximizing mask selection. Is this useful for robustness? (Adversarially robust pruning?)
