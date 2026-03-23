# Stage 2 Task: Theory Validation

Stage 1 is complete. The `pruning_core/` library is built and tested. Now validate the theoretical predictions of the Mozeika & Pizzoferrato paper.

## Context

The paper identifies four dynamical regimes based on the ratio of timescales τ_w (weight update) and τ_h (mask update):
1. Equal timescales — joint Langevin on w and h
2. Fast learning (τ_w ≪ τ_h) — Adam inner loop + Glauber outer (current implementation)
3. Fast pruning (τ_h ≪ τ_w) — masks update fast, weights lag
4. Low temperature (β→∞) — MAP limit, greedy descent (current implementation IS this)

The replica method predicts a critical ρ_c where the phase transition occurs.

## What to Build

### 2.1 — Four Regime Implementations (`pruning_core/regimes.py`)

Add to the dynamics module:

```python
def joint_langevin(w_init, h_init, X, y, eta, rho, alpha, T_w, T_h, T=100):
    """Equal timescales: simultaneous stochastic updates on w and h.
    - w update: w += -tau_w * grad_E_w + sqrt(2*T_w*tau_w) * noise
    - h update: for each j, flip h[j] with prob exp(-beta_h * max(0, delta_E))
    """

def fast_pruning(w_init, h_init, X, y, eta, rho, alpha, T=100, K_w=5):
    """Fast pruning (tau_h << tau_w): h updates many times per w update.
    - Inner loop: Glauber sweeps on h (use current w)
    - Outer loop: single Adam step on w, then many Glauber steps
    K_w: Adam steps per outer iteration (small = w lags behind h)
    """
```

The existing `run_glauber` with K=50 inner Adam steps is already the "fast learning" regime.

### 2.2 — Phase Diagram Experiment (`experiments/03_phase_diagram.py`)

This is the key experiment. For N=200, M=400 (faster than N=500):
- Sweep rho in np.linspace(0, 0.002, 25)
- Sweep eta in [0, 0.0001, 0.0005, 0.001]
- For each (eta, rho): run Glauber (fast learning regime), compute Hamming(h_final, h0)
- Fit a sigmoid to Hamming(rho) for each eta → extract critical rho_c
- Output: `results/phase_diagram.csv` with columns: eta, rho, Hamming, rho_c_estimate

Expected result: sharp drop in Hamming from ~0.5 to ~0.01 as rho crosses rho_c.

**Important performance note:** For efficiency, use N=200, M=400, T=50, K_adam=20.
This runs in ~5 minutes per cell at N=200 rather than ~20 min at N=500.

### 2.3 — Convergence Test: How many Adam steps is "enough"? (`experiments/04_adam_convergence.py`)

The "fast learning" regime assumes w is fully equilibrated before h updates.
Test: for a fixed (eta, rho) in the recovery region, sweep K ∈ {5, 10, 20, 50, 100}
- For each K, run Glauber and record final Hamming(h, h0)
- Plot: Hamming vs K to find the minimum K where results stop improving
- Output: `results/adam_convergence.csv` with columns: K, Hamming, iterations

### 2.4 — Finite-Size Scaling (`experiments/05_finite_size.py`)

Fix M/N = 2, sweep N ∈ {50, 100, 200, 500}
- For each N, run a rho sweep (20 points) at optimal eta
- Fit sigmoid slope to Hamming(rho) → measure transition sharpness
- Expected: slope increases with N (thermodynamic limit)
- Output: `results/finite_size.csv` with columns: N, rho, Hamming, sigmoid_slope

### 2.5 — Regime Comparison (`experiments/06_regime_comparison.py`)

For a single (eta, rho) in the transition region (where Hamming ~0.2):
- Run all three implemented regimes: fast_learning, fast_pruning, joint_langevin
- Measure: Hamming to ground truth, convergence speed (iterations), final energy
- Sweep T_w/T_h ratio for joint_langevin to show interpolation between regimes
- Output: `results/regime_comparison.csv`

## Implementation Notes

- Add `from scipy.optimize import curve_fit` for sigmoid fitting — install scipy in .venv
- Sigmoid model: `f(rho, rho_c, k) = 0.5 * (1 - np.tanh(k * (rho - rho_c)))`
- Use the existing `.venv` Python: `/home/petty/pruning-research/.venv/bin/python`
- Always set PYTHONPATH=/home/petty/pruning-research when running experiments
- All results go in `results/` directory (create it)
- Use small N (50-200) for speed unless the experiment specifically tests N-dependence

## Test Additions (`tests/test_regimes.py`)

- `test_joint_langevin_energy_bounded`: energy stays finite over 10 steps at small T
- `test_fast_pruning_sparse`: fast pruning at high rho produces sparser mask than no rho
- `test_phase_transition_small`: N=50, run phase diagram, verify Hamming drops from >0.3 to <0.1 as rho increases

## When Done

Run all tests, check that `results/phase_diagram.csv` shows the phase transition clearly, then:

```bash
cd /home/petty/pruning-research && git add -A && git commit -m "Stage 2: four regimes + phase diagram experiments"
openclaw system event --text "Stage 2 complete: four regimes implemented, phase transition validated numerically, finite-size scaling done" --mode now
```
