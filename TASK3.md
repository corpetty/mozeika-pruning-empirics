# Stage 3 Task: Deep Networks + Phase Diagram Fix

## Fix First: Phase Diagram Bug

The phase diagram experiment has a double-normalization bug.

In `experiments/03_phase_diagram.py`, the line:
```python
Hamming = hamming_distance(h_final.astype(int), h0.astype(int)) / N
```

The `hamming_distance` function already returns normalized values (divided by N). Remove the `/ N` division.

Also: N=200, M=400 is too easy — the algorithm is already in the recovery regime everywhere. 
Change parameters to N=200, M=100 (M/N=0.5) to actually hit the phase transition region where Hamming ~0.5 at low ρ and drops to ~0.01 at high ρ.

Re-run `experiments/03_phase_diagram.py` after the fix and verify the output shows:
- At rho=0: Hamming ≈ 0.4–0.5 (random mask region)
- At rho=0.001+: Hamming ≈ 0.01–0.05 (recovery region)
- Clear phase transition between the two

Update `results/phase_diagram.csv` with corrected output.

## Multi-Layer Extension (`experiments/07_mlp_layerwise.py`)

Architecture: 4 inputs → 8 hidden (ReLU) → 4 hidden (ReLU) → 1 output
- N_total = 4*8 + 8*4 + 4*1 = 68 weights (tractable for Glauber)
- True network: random w0, h0 with p0=0.5 per layer
- Data: M=200 samples, M_test=200

Extend the energy framework to multi-layer networks:
```python
# In pruning_core/energy.py — add:
def mlp_forward(w_list, h_list, X, activation='relu'):
    """Forward pass for MLP with masked weights."""
    
def mlp_loss(w_list, h_list, X, y, activation='relu'):
    """Squared loss for MLP."""
    
def mlp_total_energy(w_list, h_list, X, y, eta_list, alpha, rho_list):
    """Total energy summed over layers: Σ_l [L + η_l||w_l||² + ρ_l Σ V(h_{l,i})]"""
    
def mlp_grad_w(w_list, h_list, X, y, eta_list, activation='relu'):
    """Gradients ∂E/∂w_l for each layer via backprop."""
```

Implement Glauber for MLPs in `pruning_core/dynamics.py`:
```python
def run_glauber_mlp(w_init_list, h_init_list, X, y, eta_list, rho_list, alpha, T=50):
    """Glauber dynamics for multi-layer network.
    - For each layer l, sweep coordinates in random order
    - Flip h[l][j] if ΔE < 0 (same MAP rule as perceptron)
    - After each full layer sweep, re-optimize all w via Adam
    """
```

Experiment:
- Run layerwise Glauber at several (η, ρ) values
- Measure per-layer sparsity and Hamming(h_l, h0_l) for each layer
- Key question: does the phase transition exist per-layer independently?
- Output: `results/mlp_layerwise.csv` with columns: eta, rho, layer, Hamming, sparsity

## Layer Collapse Analysis (`experiments/08_layer_collapse.py`)

Using the same 4→8→4→1 MLP:
- Fix η=0.0001, sweep ρ from 0 to 0.01 (wider range than perceptron)
- For each ρ: run Glauber, measure fraction of active neurons per layer
  - A neuron is "dead" if all its incoming weights are masked to 0
- Output: `results/layer_collapse.csv` with columns: rho, layer, active_fraction, Hamming
- Expected: one layer collapses before others as ρ increases — which one?

## Activation Comparison (`experiments/09_activation_comparison.py`)

Compare identity (linear), tanh, and ReLU activations on the perceptron:
- N=100, M=200, sweep ρ at fixed η=0.0005
- For each activation: run Glauber, measure Hamming and convergence iterations
- Note: for non-identity activations, gradients change (use autograd-style finite diff or implement tanh grad)
- Output: `results/activation_comparison.csv` with columns: activation, rho, Hamming, iterations

For gradients with tanh: ∂L/∂w_j = (1/M) Σ_i (φ(X_i · (w∘h)) - y_i) · φ'(X_i · (w∘h)) · h_j · X_{ij}
where φ'(x) = 1 - tanh²(x)

For ReLU: φ'(x) = 1 if x > 0 else 0

## Test Additions (`tests/test_mlp.py`)

- `test_mlp_forward_shape`: forward pass output shape correct
- `test_mlp_energy_nonneg`: energy ≥ 0 for valid inputs
- `test_mlp_glauber_decreasing`: energy non-increasing over Glauber steps
- `test_layer_collapse_detection`: detect dead neurons correctly

## When Done

Run all tests, verify phase_diagram.csv shows the actual phase transition (Hamming drops from ~0.5 to ~0.01), then:

```bash
cd /home/petty/pruning-research && git add -A && git commit -m "Stage 3: MLP extension, layer collapse, activation comparison, phase diagram fix"
openclaw system event --text "Stage 3 complete: MLP multi-layer Glauber, layer collapse analysis, activation comparison, phase diagram fixed" --mode now
```
