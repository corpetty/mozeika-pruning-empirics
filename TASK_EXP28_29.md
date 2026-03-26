# Task: Experiments 28 & 29 — Correct Mozeika Validation

## Context

Previous experiments (17-27) tested the wrong things. Key corrections:

1. **kappa is the Hessian diagonal** (Fisher is an approximation, valid at well-specified optima)
2. **The algorithm is OBD, not Glauber on synthetic data** — replicas/Langevin are the *analysis*; OBD is the *algorithm*
3. **Algorithm order is mandatory**: fully train to local minimum → estimate curvature → prune → fine-tune → repeat. Scoring during training gives garbage saliency.
4. **Phase transition in MLPs is NOT predicted by the paper** — we tested a strawman
5. **For synthetic mask recovery: use LINEAR activations** — ReLU + permutation symmetry makes h₀ recovery ill-posed

## Repo & Environment

- Repo: `/home/petty/pruning-research` (git, Python 3.12 venv at `venv/`, numpy/scipy/sklearn)
- **PyTorch venv** (for exp 29 MNIST): `/home/petty/torch-env/bin/python3` (torch 2.11+cu128, both RTX 3090s)
- Reference: `lenet300_pruning.py` is Mozeika's correct implementation — read it before writing exp 29

## Tool call parameter names (CRITICAL)

When using the Edit tool, use `old_string` and `new_string` (NOT old_text/new_text/oldText/newText).
When using Write tool, use `path` and `content`.

---

## Experiment 28: Synthetic Teacher, Linear Activation

**File:** `experiments/28_synthetic_linear.py`
**Python:** `venv/bin/python3` (numpy/scipy only, no torch needed)

**Goal:** Validate that OBD saliency correctly recovers a sparse teacher mask when:
- Data is exactly Gaussian
- Network has linear activations (no ReLU — eliminates permutation symmetry)
- Algorithm follows the correct train→score→prune→finetune cycle

**Architecture:**
```
N_in = 100, N_h = 50, N_out = 1
Teacher: sparse W1 (N_h x N_in), sparse W2 (N_out x N_h), linear activation
y = W2 @ W1 @ x + noise (small noise sigma=0.01)
Teacher sparsity p0 = 0.5 (50% zeros in each weight matrix)
M_train = 5000 samples
```

**Algorithm (correct cycle):**
```
For each rho in rho_grid (e.g., np.logspace(-6, -1, 20)):
    For each seed in range(5):
        # Initialize student with same architecture, all masks active
        W1 = random init, H1 = ones (all active)
        W2 = random init, H2 = ones (all active)
        
        For iteration in range(max_iter=30):
            # Step 1: Fully train weights to local minimum (masks fixed)
            Run Adam for K=500 steps minimizing:
                L = ||y - W2 @ diag(H2) @ (W1 @ diag(H1)) @ X.T||^2 / (2M) + eta * (||W1||^2 + ||W2||^2)
            where eta = 0.001
            
            # Check convergence: if grad norm < 1e-5, stop early
            
            # Step 2: Compute OBD saliency scores (after convergence)
            For each active weight w_ij in W1:
                F_ij = mean of g_ij^2 over training data (diagonal Fisher proxy)
                S_ij = 0.5 * F_ij * w_ij^2
            Same for W2
            
            # Step 3: Prune (update masks)
            H1[S1 < rho/2] = 0  (prune active weights with low saliency)
            H2[S2 < rho/2] = 0
            
            # Dead neuron cleanup: if all H1[:, k] == 0 for some neuron k,
            # also set H2[0, k] = 0 (neuron k is dead)
            
            # If no weights changed, stop (converged)
        
        # Measure mask recovery
        # NOTE: With linear activations (no ReLU), permutation ambiguity still exists
        # but W2 @ W1 is a rank-deficient product — the correct comparison is
        # Hamming distance on the COMBINED effective weight W_eff = W2 * W1
        # or use Hungarian matching on layer 1 mask only
        
        ham1 = hamming(H1.flatten(), H1_teacher.flatten())
        ham2 = hamming(H2.flatten(), H2_teacher.flatten())
        active_frac1 = H1.mean()
        active_frac2 = H2.mean()
        
        Record: rho, seed, ham1, ham2, active_frac1, active_frac2
```

**IMPORTANT NOTE on identifiability:**
Even with linear activations, a 2-layer linear network has the form y = W2 W1 x, so W2 W1 can be factored many ways. The mask is NOT uniquely identifiable in general unless the network is single-layer. Two options:
1. **Single hidden layer only**: set N_h = 1 effectively (make it a 2-weight linear model) — too trivial
2. **Single layer with input selection**: set W2 = ones (dense, frozen), only learn W1 — then W1 mask IS identifiable as in the perceptron
3. **Just measure recovery of W_eff = W2 @ W1**: check if the effective map converges to the teacher's W2_true @ W1_true

Recommended: option 2 (freeze W2=random fixed, only mask W1). This is closest to the single-layer perceptron case and gives a clean mask recovery test.

**Save results to:** `results/synthetic_linear.csv`
Columns: rho, seed, ham1, active_frac1, n_iter_converged

**After running**, plot: Hamming distance vs rho (log scale). Should see sharp drop at some rho_c.

---

## Experiment 29: OBD vs Magnitude on LeNet-300-100 + MNIST

**File:** `experiments/29_obd_vs_magnitude_mnist.py`
**Python:** `/home/petty/torch-env/bin/python3` (PyTorch + CUDA)

**Goal:** The CANONICAL comparison from OBD literature: does curvature-based saliency beat magnitude pruning at high sparsity on a real network?

**Reference:** Read `lenet300_pruning.py` first — this is Mozeika's correct implementation. Copy its `MaskedLinear`, `MaskedLeNet`, and Fisher estimation code.

**Architecture:** LeNet-300-100 (784 → 300 → 100 → 10), MNIST, CrossEntropy loss

**Layer-specific rho values** (from lenet300_pruning.py):
```python
rho_config = {
    'fc1': {'weight': rho_fc1_w, 'bias': rho_fc1_b},
    'fc2': {'weight': rho_fc2_w, 'bias': rho_fc2_b},
    'fc3': {'weight': rho_fc3_w, 'bias': rho_fc3_b},
}
```

**Two methods to compare:**

**Method A — OBD/Mozeika saliency:**
1. Pretrain full network (no masks) for 10 epochs
2. Repeat 5 rounds of: estimate Fisher diagonal (500 batches) → prune (S_i < rho/2) → fine-tune 5 epochs
3. Measure test accuracy at each round

**Method B — Magnitude pruning:**
1. Pretrain full network for 10 epochs (same as A)
2. For each target sparsity level matching Method A's achieved sparsity: prune globally by |w_i| threshold → fine-tune 5 epochs
3. Measure test accuracy

**Sparsity levels to target:** 50%, 70%, 85%, 90%, 95%

**Key metric:** Test accuracy at matched sparsity. Expected: OBD wins at ≥85% sparsity because it avoids removing high-curvature near-zero weights.

**Save results to:** `results/obd_vs_magnitude_mnist.csv`
Columns: method, sparsity, test_acc, round

---

## Deliverables

1. `experiments/28_synthetic_linear.py` — clean, commented, uses numpy only
2. `experiments/29_obd_vs_magnitude_mnist.py` — uses torch-env, reads lenet300_pruning.py for reference
3. `results/synthetic_linear.csv` — exp 28 results
4. `results/obd_vs_magnitude_mnist.csv` — exp 29 results
5. Both experiments committed to git

## Success criteria

- Exp 28: Hamming distance shows clear drop toward 0 as rho increases past rho_c. If Hamming stays ~0.5, the identifiability approach needs revision.
- Exp 29: OBD accuracy ≥ magnitude accuracy at 85%+ sparsity. If magnitude wins at all sparsities, that itself is an interesting result worth reporting.

## After completion

Update `/home/petty/pruning-research/CORRECTION_PLAN.md` checkboxes and write a brief summary of what was found to `results/exp28_29_summary.txt`.
