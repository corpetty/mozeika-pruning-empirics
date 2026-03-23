# Stage 5 & 6 Task Brief

## CRITICAL: Tool Call Rules
Use ONLY these exact parameter names or calls will silently fail:

- **exec**: `command` (string)
- **Read**: `path` (string)
- **Write**: `path` (string), `content` (string)
- **Edit**: `path` (string), `old_string` (string), `new_string` (string)

All python must run via: `cd /home/petty/pruning-research && venv/bin/python <script>`
Run tests with: `cd /home/petty/pruning-research && venv/bin/python -m pytest tests/ -q`

## Environment
- Repo: /home/petty/pruning-research
- Python: /home/petty/pruning-research/venv/bin/python
- Packages: numpy, scipy, pytest (in venv)
- Git: commit after each stage completes

## Current State (HEAD: cad055d)
Stages 1-4 complete. 17/17 tests passing.

**Recent bug fixed:** `grad_mlp_loss_w` in `pruning_core/energy_mlp.py` was using raw `w` in forward pass instead of `w_masked`, and didn't handle 1D (perceptron) vs 2D (MLP) weight shapes. Now fixed.

**Files in place:**
- pruning_core/: energy.py, energy_mlp.py, dynamics.py, optimizers.py, data.py, metrics.py, regimes.py, replicas.py
- experiments/: 01-12, 14 (replica draft)
- results/: phase_diagram.csv, variance_concentration.csv, subspace_angles.csv, others

## Stage 5: Rényi/Replica Knob
Goal: Validate that n = beta_h/beta_w in multi-replica Glauber acts as the Rényi order parameter.
Prediction: n=1 → standard Bayesian (smooth recovery), n→∞ → minimax mask selection (sharp threshold).

### 5a — Fix experiment 14 and run it
File `experiments/14_replica_comparison.py` exists but has a copy of MultiReplicaGlauber inlined.
Instead: import from `pruning_core/replicas.py` which has the canonical implementation.

The experiment should:
- Sweep n in [1, 2, 4, 8] and rho in linspace(0, 0.002, 11)
- For each (n, rho): run MultiReplicaGlauber on a small perceptron (N=100, M=300) for T=30
- Metric: Hamming distance to true mask h0
- Save to results/replica_comparison.csv: n, rho, hamming_mean, hamming_std (5 seeds)

Expected result: higher n → sharper phase transition (lower rho_c, steeper drop).

### 5b — rho_c extraction vs replica prediction
- From phase_diagram.csv (rho, hamming) already on disk, extract empirical rho_c as the rho where hamming drops below 0.1
- From Mozeika theory: rho_c ≈ 2*sqrt(alpha * eta) in the single-replica case
- Compare empirical vs theoretical for the alpha/eta values used in experiment 03
- Save comparison to results/rho_c_comparison.csv: source, rho_c, alpha, eta

## Stage 6: Practical Pruner
Goal: A usable `pruning_core/pruner.py` module that wraps everything into a clean API.

```python
from pruning_core.pruner import GlauberPruner

pruner = GlauberPruner(rho=0.001, eta=0.0001, n_replicas=1)
pruner.fit(X_train, y_train, layer_sizes=[10, 8, 1])
mask = pruner.get_mask()          # list of binary arrays
sparsity = pruner.sparsity()      # float, fraction of zeros
pred = pruner.predict(X_test)     # forward pass with pruned weights
```

Internally uses Glauber dynamics from `pruning_core/dynamics.py` or `replicas.py`.
Add 3 tests in `tests/test_pruner.py`: fit runs, mask is binary, sparsity is in [0,1].

## Deliverables
1. results/replica_comparison.csv
2. results/rho_c_comparison.csv
3. pruning_core/pruner.py
4. tests/test_pruner.py
5. All 20+ tests passing
6. Git commit: "Stage 5-6 complete: replica knob, rho_c comparison, GlauberPruner API"

## When done
Output a brief summary: what the replica sweep showed, what the rho_c comparison found, and confirm tests pass + commit SHA.
Output ONLY the summary as your final message — do not use the message tool to send it.
