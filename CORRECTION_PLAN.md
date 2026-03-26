# Correction Plan: What We Got Wrong and How to Fix It

*Updated: 2026-03-25 — based on Mozeika's code, his ChatGPT reasoning session, and his feedback that the "Interpretation and Where This Points" section is "most of this is incorrect"*

---

## What We Got Wrong

### 1. We misidentified kappa (the curvature term)

**What we said:** kappa ≈ Fisher information matrix diagonal.

**What it actually is:** kappa = H_ii — the diagonal of the Hessian (second derivative of the loss). Fisher and Hessian are only equal under specific conditions: negative log-likelihood loss, well-specified model, expectations over the true data-generating distribution. In practice, empirical Fisher ≠ empirical Hessian. We used diagonal Fisher as a proxy and didn't distinguish these clearly.

**Implication for our code:** Our `estimate_diag_fisher()` function computes E[g²] (outer product of gradients), which approximates the Fisher, not the Hessian. This is a valid approximation but we should be explicit about what we're computing and when it diverges from the true curvature.

---

### 2. We conflated the statistical mechanics framework with the practical algorithm

**What we said:** The Langevin dynamics, temperature parameters, and replica framework *are* the algorithm. We tested all four "regimes" as if they were competing implementations of the same thing.

**What it actually is:** The statistical mechanics machinery (replicas, temperatures, thermodynamic limits) is an *analysis* framework — it computes typical properties of ensembles of solutions and derives why quadratic saliency scores emerge. The actual practical algorithm it motivates is the **OBD/OBS family**: rank parameters by S_i ≈ ½ H_ii w_i², prune the bottom ones, fine-tune.

The "fast learning" regime doesn't mean "run Adam fast and then do Glauber" — it means in the asymptotic limit where weights equilibrate faster than masks, the effective marginal distribution over masks reduces to something tractable (Bayesian model selection). That's a theoretical result, not an algorithmic step.

**What this means for our experiments:** Experiments 2–6 (four regime comparison) were testing something reasonable but not what the paper prescribes for practitioners. The paper says: compute OBD saliency, prune, fine-tune. The regimes are theoretical limits that explain *why* this works, not competing algorithms.

---

### 3. The saliency score formula was partially wrong

**What we implemented:**
```python
sal_w = 0.5 * fisher[f"{name}.weight"] * (eff_w ** 2)
```

This is `S_i ≈ ½ F_ii w_i²` — OBD with Fisher instead of Hessian. This is a common approximation (used in many pruning papers) but:
- It ignores the sparsity reward term: **S_i should include the ρ contribution** when deciding whether to flip.
- The full score for the pruning decision is: `ΔE_i = ½ H_ii w_i² - ρ` (gain from sparsity minus cost from loss increase)
- We prune if `½ H_ii w_i² < ρ`, i.e., the saliency is *below* ρ — which is what our code does, but the ρ threshold is exactly the sparsity pressure parameter, not a separate hyperparameter.

In Mozeika's LeNet code (`lenet300_pruning.py`), the pruning condition is:
```python
candidate_w = active_w & (sal_w < (rho_w / 2.0))
```

So `prune if S_i < ρ/2`. This is the correct test. **Our Glauber implementation was doing greedy coordinate descent on the binary mask — which is the MAP T→0 limit — but the saliency score itself (OBD with Fisher proxy) was reasonable, just not identical to what the paper derives.**

---

### 4. The "mean-field breakdown" interpretation is wrong (or at least premature)

**What we said:** The Mozeika framework is mean-field theory. It breaks down in MLPs because hidden layers create sparse factor graphs where mean-field fails.

**What Mozeika likely intended:** The framework does *not* claim the OBD saliency rule works perfectly on arbitrary MLPs. It claims that:
1. The saliency score S_i = ½ H_ii w_i² is the theoretically grounded way to measure the importance of a weight
2. The pruning rule (prune if S_i < ρ) is optimal in the limit where the fast-learning regime holds
3. For real multi-layer networks, you apply the same saliency score layer-by-layer with layer-specific ρ values

The paper is not claiming a sharp phase transition in MLPs. It's deriving the *correct curvature-based saliency* and justifying why it's better than magnitude. The "phase transition" in the perceptron is a mathematical property of that specific model, not a claim about neural networks in general.

**So our "MLP phase transition" experiments (Exp 20) were testing a strawman.** The paper doesn't predict a sharp Hamming-distance transition in MLPs. It predicts that curvature-weighted saliency will give better pruning than magnitude. That's a different and more modest claim.

---

### 5. Our "practical advantage" framing was also off

We framed Exp 25 as "does Mozeika beat L1?" The correct framing is "does OBD-style curvature saliency find better sparse networks than magnitude, which ignores curvature?" The comparison should be:

- **Magnitude pruning** (ignores curvature): sort by |w_i|, prune smallest
- **OBD/Mozeika saliency** (uses curvature): sort by ½ H_ii w_i², prune smallest
- At high sparsity, magnitude makes catastrophically bad choices (removes near-zero weights that have high curvature); OBD avoids this

This comparison (magnitude vs OBD) is the canonical result from OBD papers (LeCun et al. 1990, Hassibi & Stork 1993) and it's well-established that OBD wins. We didn't test the right thing.

---

## What Mozeika Actually Built

The `lenet300_pruning.py` code is correct Mozeika-framework implementation:

1. **MaskedLinear layers** — binary masks applied as Hadamard product on weights and biases ✅
2. **Saliency score** = `0.5 * F_ii * w_i²` (OBD with diagonal Fisher proxy) ✅
3. **Pruning condition** = `S_i < ρ/2` per layer with separate ρ_w, ρ_b per layer ✅
4. **Training loop**: pretrain → (estimate Fisher → prune → fine-tune) × rounds ✅
5. **Dead neuron cleanup** — if all incoming weights pruned and φ(0)=0, prune outgoing too ✅
6. **Layer-specific ρ values**: different ρ for fc1, fc2, fc3 weights, and for biases ✅

This is the paper's prescribed algorithm applied properly to LeNet-300-100.

---

## Mozeika's Suggested Next Step (Synthetic Teacher Network)

His suggestion is the right one for validating the framework cleanly:

**Goal:** Remove confounds from real data (non-Gaussian inputs, label noise, complex loss landscape). Test whether the OBD saliency correctly identifies the true mask in a controlled setting where ground truth is known.

**Protocol:**
1. **Generate a sparse teacher network**: random architecture with one hidden layer, sparse random weight initialization with known mask h₀
2. **Generate dataset**: feed i.i.d. Gaussian inputs X ~ N(0, I), compute outputs y = f_teacher(X) — this is synthetic regression with known ground truth
3. **Train a student network** with the same architecture using the fast-learning algorithm (zero temperature: optimize w for fixed h, then update h by OBD saliency rule)
4. **Measure**: does the recovered mask match h₀? Plot Hamming distance between recovered h and true h₀ as a function of ρ. Look for the phase transition.

**Why this is better than what we did:**
- Ground truth mask is known (unlike MNIST where the "correct" sparse network is ambiguous)
- The data is exactly Gaussian and uncorrelated (assumption in the theory)
- The teacher network generates the labels so the student can in principle recover h₀ exactly
- This is the exact setup the perceptron experiments used, extended to a multi-layer teacher

---

## Implementation Plan

### Phase 0: Fix the understanding (no code yet)

Before writing any new code, we need to be clear on:
- The algorithm we're implementing is **OBD-style curvature saliency** with Fisher proxy for H
- The test is **mask recovery on a synthetic teacher** (not accuracy on MNIST)
- "Phase transition" in MLPs is not predicted — the question is "does OBD recover the teacher mask?"
- ρ is not found analytically — it's swept over a grid or tuned to get target sparsity

### Phase 1: Synthetic teacher experiment (implements Mozeika's suggestion)

**Architecture:**
```python
# Teacher: sparse single hidden layer
N_in = 100      # input dimension
N_h = 50        # hidden neurons
N_out = 1       # regression output
p0 = 0.3        # sparsity of teacher (70% of weights are zero)
M = 5000        # training samples

# Teacher generation:
W1_true = np.random.randn(N_h, N_in) * (np.random.rand(N_h, N_in) > p0)
W2_true = np.random.randn(N_out, N_h) * (np.random.rand(N_out, N_h) > p0)

# Dataset:
X = np.random.randn(M, N_in)
z1 = np.maximum(X @ W1_true.T, 0)  # ReLU hidden layer
y = z1 @ W2_true.T  # linear output
```

**Student training (zero-temperature fast-learning)**:
```
For ρ in rho_grid:
    Initialize student weights randomly (same architecture as teacher)
    Initialize all masks to 1 (all active)
    
    Repeat until convergence:
        # Fast-learning inner loop (w update with h fixed)
        Train student to local minimum with Adam, L2 penalty η
        
        # OBD pruning step (h update with w fixed)
        Compute diagonal Fisher: F_ii = E[g_i²]  
        Compute saliency: S_i = 0.5 * F_ii * w_i²
        Prune all active weights where S_i < ρ/2
        
    Measure: Hamming(h_recovered, h_teacher)
```

**Expected result:** Sharp drop in Hamming distance at some ρ_c, similar to the perceptron case. This would confirm the theory extends to MLPs in the synthetic setting.

### Phase 2: Correct the existing experiments

Update `pruning_core/pruner.py` to:
1. Rename `fisher` to `curvature_proxy` (be honest about what it is)
2. Add a flag `use_hessian_diagonal=False` — if True, compute true Hessian diagonal via Hessian-vector products instead of Fisher proxy
3. Fix saliency to include ρ in the pruning decision rather than as a separate threshold

### Phase 3: Correct LeNet-300-100 experiment

Replace Experiment 22 with a proper OBD-vs-magnitude comparison:
- Same model (LeNet-300-100), same dataset (MNIST)
- Compare: magnitude pruning vs OBD saliency, at matched sparsity levels
- The correct comparison: after pruning, fine-tune both, measure test accuracy
- Expected result per OBD literature: OBD wins at high sparsity because it avoids removing high-curvature near-zero weights

### Phase 4: Updated MEETING_REPORT

Rewrite the "Interpretation and Where This Points" section:
- Remove the "mean-field breakdown" framing (it was not what the paper predicts)
- Replace with: "we tested the wrong things; here's what the theory actually predicts and here's new data"
- Include Phase 1 and Phase 3 results

---

## Summary of Errors vs. Correct Understanding

| What We Said | What's Correct |
|---|---|
| kappa = Fisher | kappa = Hessian; Fisher is an approximation |
| Langevin/replicas/temps = the algorithm | They're the *analysis*; OBD is the algorithm |
| Paper predicts phase transition in MLPs | Paper doesn't predict this; perceptron transition is a special case |
| "Mean-field breaks down" explains our MLP results | We tested the wrong thing (Glauber on synthetic data, not OBD on teacher network) |
| No practical advantage over L1 | We didn't test OBD vs magnitude (the correct comparison) |
| Framework fundamentally limited | Actually: we didn't test what it claims; start over with teacher network |

---

## Critical Fixes from Branch PDF (2026-03-25)

Two corrections from the ChatGPT conversation Mozeika shared:

**Fix 1 — Algorithm order is mandatory:**
The correct cycle is: **(1) fully train weights to local minimum → (2) estimate diagonal Fisher/Hessian → (3) prune all weights where S_i < ρ/2 → (4) fine-tune → repeat.**
Experiment 27 was computing OBD scores *during* training, not after convergence. At a non-minimum, gradients are nonzero and Fisher ≠ curvature of the loss, so saliency scores are garbage.

**Fix 2 — Linear activations required for synthetic teacher recovery:**
With ReLU activations, permutation symmetry + sign ambiguity make h₀ recovery ill-posed for multi-layer networks. With **linear activations**, the network collapses to a linear map and the sparse decomposition is identifiable (up to the Kruskal rank condition). The synthetic teacher experiment must use linear activations. Mozeika's perceptron experiments were effectively doing this (single layer = trivially linear path from input to output).

## Next Actions (revised 2026-03-25)

1. **[ ]** Implement exp 28: synthetic teacher, LINEAR activation, proper train→score→prune→finetune cycle (`experiments/28_synthetic_linear.py`)
2. **[ ]** Implement exp 29: OBD vs magnitude on LeNet-300-100 + MNIST at matched sparsity levels (`experiments/29_obd_vs_magnitude_mnist.py`) — use PyTorch venv at `/home/petty/torch-env`
3. **[ ]** Rewrite MEETING_REPORT.md "Interpretation" section
4. **[ ]** Talk to Mozeika directly with corrected framing

---

*This plan was written after reviewing: (1) Mozeika's `lenet300_pruning.py` — the correct implementation of his framework; (2) his ChatGPT reasoning session on LeNet-300-100 vs CNN, which clarifies OBD/OBS saliency, kappa definition, and the role of the stat-mech analysis; (3) his feedback that our interpretation section was "most of this is incorrect".*
