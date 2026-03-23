# Stage 1 Task: Python Baseline Implementation

You are implementing Stage 1 of a research codebase for the paper "Statistical Mechanics of Learning and Pruning in Neural Networks" (Mozeika & Pizzoferrato, 2026).

## Context

The paper frames neural network pruning as coupled stochastic Langevin dynamics over weights w and binary masks h. The key energy function is:

    E(w, h | D) = L(w∘h | D) + (η/2)||w||² + Σ V(hᵢ)
    
where V(h) = α·h²(h-1)² + (ρ/2)·h is a double-well potential.

- η = L2 regularization on weights
- ρ = sparsity pressure (higher ρ → more pruning)
- α = double-well barrier height
- w∘h = Hadamard product (elementwise)
- L(w) = squared loss

There is a phase transition: below a critical ρ_c the algorithm fails to recover the true mask h₀; above it, Hamming distance to h₀ drops sharply.

## Reference Implementation

`perceptron_pruning_v5.1.r` in this directory is the working R implementation. Key details:
- φ(x) = x (identity activation, linear perceptron)
- Adam optimizer for ∂E/∂w (inner loop, K=50 steps)
- Glauber dynamics for h (outer coordinate search, flip h[j] if ΔE < 0)
- This is the **low-temperature MAP limit** (β→∞): only downhill moves accepted
- `9900_stats.csv` contains reference output: 11×11 grid of (η, ρ) values, columns: eta, rho, it, sum_h, Hamming, MSE, E, train_error, test_error

## What to Build

### 1. Project structure
```
pruning-research/
├── pruning_core/
│   ├── __init__.py
│   ├── energy.py       # E(w,h|D), L(w), V(h), gradients
│   ├── optimizers.py   # Adam implementation
│   ├── dynamics.py     # Glauber, exhaustive_search
│   ├── data.py         # synthetic data generators
│   └── metrics.py      # Hamming, MSE, sparsity helpers
├── experiments/
│   ├── __init__.py
│   ├── 01_perceptron_glauber.py   # reproduce 11×11 grid
│   └── 02_nn_exhaustive.py        # 4→3→1 net, 2^N enumeration
├── tests/
│   ├── test_energy.py
│   ├── test_optimizers.py
│   └── test_dynamics.py
├── requirements.txt
└── README.md
```

### 2. `pruning_core/energy.py`
- `squared_loss(w, h, X, y, phi=None)` — L(w∘h | D)
- `double_well(h, alpha, rho)` — V(h) vectorized
- `total_energy(w, h, X, y, eta, alpha, rho)` — full E(w,h|D)
- `grad_energy_w(w, h, X, y, eta, phi=None)` — ∂E/∂w (identity and tanh)

### 3. `pruning_core/optimizers.py`
- `AdamOptimizer` class matching the R implementation exactly (same defaults: lr=1e-2, β1=0.9, β2=0.999, ε=1e-8)
- `optimize_w(w_init, h, X, y, eta, K=50, lr=1e-2)` — run K Adam steps, return optimized w

### 4. `pruning_core/dynamics.py`
- `glauber_step(w, h, X, y, eta, rho, alpha)` — one full sweep over N coordinates (random order), flip if ΔE < 0
- `run_glauber(w_init, h_init, X, y, eta, rho, alpha, T=100)` — iterate until convergence or T steps
- `exhaustive_search(X, y, eta, rho, alpha, N, K_adam=50)` — enumerate all 2^N masks, return best (w, h, loss)

### 5. `pruning_core/data.py`
- `sample_perceptron(N, M, p0, sigma=0.01, seed=None)` — returns X, y, w0, h0 matching R exactly:
  - w0 ~ N(0, I_N)
  - h0: exactly floor(N*p0) ones, rest zeros, randomly placed
  - X ~ N(0, I_N/N) (note the 1/√N scaling in R: `mvrnorm(M, zero, D_N)/sqrt(N)`)
  - y = X @ (w0*h0) + sigma*noise

### 6. `experiments/01_perceptron_glauber.py`
- Reproduce the 11×11 grid: eta ∈ linspace(0, 0.001, 11), rho ∈ linspace(0, 0.001, 11)
- N=500, p0=0.5, M=1000, M_test=1000, T=100, seed=9900
- Output CSV with same columns as 9900_stats.csv
- Print progress like R code does

### 7. `experiments/02_nn_exhaustive.py`
- Architecture: 4 inputs → [4,4,2] hidden neurons → 1 output (or similar small N ≤ 20)
- True network: sample random w0, h0, generate data
- Exhaustive search over all 2^N masks for each (η, ρ) in a small grid
- Output: heatmap data (η, ρ, min_loss, Hamming(h_min, h0))

### 8. `tests/`
- `test_energy.py`: verify E(w, ones, X, y, eta=0, rho=0) == L(w); verify V(0)=V(1)=0; verify V(0.5)>0
- `test_optimizers.py`: Adam on simple quadratic converges in <100 steps
- `test_dynamics.py`: Glauber on tiny N=5 example, energy is non-increasing

## Validation Target

Run `experiments/01_perceptron_glauber.py` with seed=9900. The output CSV should show:
- At rho=0, eta=0: Hamming ≈ 0.45 (random mask region)  
- At rho≥0.0007, eta≥0.0001: Hamming ≈ 0.008 (recovery region)
- Phase transition between rho=0.0004 and rho=0.0007

Exact numbers will differ by seed variation in the Python port, but the qualitative structure must match.

## Implementation Notes

- Use numpy only (no PyTorch yet — that's Stage 4)
- Match R's MASS::mvrnorm behavior: use `np.random.multivariate_normal`
- The R code uses 1-indexed arrays; be careful with off-by-one in the Python port
- `think: false` if using any local LLM calls — not needed here, just numpy

## When Done

Run the tests, run experiment 01, verify the CSV looks right, then run:
```bash
openclaw system event --text "Stage 1 complete: Python baseline done, experiment 01 validated against R reference" --mode now
```

Commit everything to git before sending that event.
