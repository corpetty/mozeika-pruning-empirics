# Statistical Mechanics of Learning and Pruning in Neural Networks

Reference implementation of the paper: **"Statistical Mechanics of Learning and Pruning in Neural Networks"** (Mozeika & Pizzoferrato, 2026)

## Overview

This project frames neural network pruning as coupled stochastic Langevin dynamics over weights `w` and binary masks `h`. The energy function is:

```
E(w, h | D) = L(w ∘ h | D) + (η/2)||w||² + Σ V(hᵢ)
```

where `V(h) = α·h²(h-1)² + (ρ/2)·h` is a double-well potential.

### Key Results

- There is a **phase transition** at critical sparsity pressure `ρ_c`
- Below `ρ_c`: algorithm fails to recover true mask `h₀`
- Above `ρ_c`: Hamming distance to `h₀` drops sharply

## Structure

```
pruning-research/
├── pruning_core/        # Core algorithms
│   ├── __init__.py
│   ├── energy.py        # Energy functions and gradients
│   ├── optimizers.py    # Adam optimizer
│   ├── dynamics.py      # Glauber dynamics, exhaustive search
│   ├── data.py          # Synthetic data generators
│   └── metrics.py       # Evaluation metrics
├── experiments/         # Experiments
│   ├── __init__.py
│   ├── 01_perceptron_glauber.py  # 11×11 grid reproduction
│   └── 02_nn_exhaustive.py      # Small net exhaustive search
├── tests/              # Unit tests
│   ├── test_energy.py
│   ├── test_optimizers.py
│   └── test_dynamics.py
├── requirements.txt
├── README.md
└── TASK.md             # Stage 1 task specification
```

## Installation

```bash
cd /home/petty/pruning-research
pip install -r requirements.txt
```

## Running Experiments

### Experiment 1: Perceptron with Glauber Dynamics

Reproduces the 11×11 grid from the R implementation:

```bash
python experiments/01_perceptron_glauber.py
```

Output: `9900_stats.csv` (same format as R reference)

### Experiment 2: Network Exhaustive Search

Exhaustive enumeration over all `2^N` masks for small networks:

```bash
python experiments/02_nn_exhaustive.py
```

Output: `exhaustive_search_results.json`

## Running Tests

```bash
python -m pytest tests/ -v
```

## Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `η` (eta) | L2 regularization on weights | 0 → 0.001 |
| `ρ` (rho) | Sparsity pressure | 0 → 0.001 |
| `α` (alpha) | Double-well barrier height | 0.5 → 2.0 |

## Validation Targets

- **Phase transition:** Hamming distance should drop sharply between `ρ=0.0004` and `ρ=0.0007`
- **At ρ=0, η=0:** Hamming ≈ 0.45 (random mask region)
- **At ρ≥0.0007, η≥0.0001:** Hamming ≈ 0.008 (recovery region)

## Reference Implementation

The R code in `perceptron_pruning_v5.1.r` is the ground truth. Key details:
- **φ(x) = x**: identity activation (linear perceptron)
- **Adam optimizer**: K=50 steps, lr=1e-2
- **Glauber dynamics**: coordinate descent with random order
- **Low-temperature limit**: only downhill moves accepted

## Next Steps (Stage 2+)

1. **PyTorch implementation**: Replace numpy with PyTorch for GPU acceleration
2. **Deeper networks**: Multi-layer perceptrons and convolutional nets
3. **Stochastic dynamics**: Add noise terms for finite temperature
4. **Experimental verification**: Compare with real pruning methods

## License

See LICENSE file for licensing details.
