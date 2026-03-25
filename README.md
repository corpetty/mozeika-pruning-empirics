# Mozeika-Pruning Empirics

**Empirical evaluation of the Mozeika & Pizzoferrato (2026) statistical mechanics pruning framework.**

This repo contains 25 experiments, a full Python implementation of the theoretical framework, all result data, an interactive visualization, and a meeting-ready report. The short version: the phase transition predicted by the theory is real and sharp in linear perceptrons, but does not generalize to MLPs, CNNs, or real networks. L1 regularization captures the useful behavior more efficiently.

---

## Key Findings

| Experiment | Finding |
|---|---|
| Linear perceptron (Exp 1–16) | Sharp phase transition confirmed at ρ_c ≈ 0.0001 (N=60, σ=0.01) |
| Rényi replica sharpening (Exp 17, 23) | Null — thermal noise O(1) swamps energy differences O(σ²/N) ≈ 10⁻⁵. MAP is already optimal. |
| ρ_c prediction (Exp 18, 24) | Mozeika formula 2√(αη) off by 100–20,000×. Empirical fit only works ~40% of cases. |
| MLP phase transition (Exp 20) | No sharp transition. Hamming stays ~0.46–0.51 regardless of ρ. |
| CNN/MNIST (Exp 22) | Magnitude pruning beats Mozeika post-finetune at all sparsity levels. |
| Definitive baseline (Exp 25) | L1 regularization beats Mozeika at 50–75% sparsity. Mozeika beats magnitude+retrain above 85%, but L1 still wins overall. No regime where Mozeika beats L1. |

**Theoretical framing:** The phase transition is a mean-field artifact. Linear perceptrons have dense interaction graphs where mean-field is exact. Hidden layers create sparse, layered factor graphs — exactly where mean-field breaks down. Natural extension: belief propagation (Krzakala/Zdeborová 2019 direction).

---

## Repo Structure

```
pruning-research/
├── pruning_core/           # Core library
│   ├── energy.py           # Energy functions and gradients (perceptron + MLP)
│   ├── optimizers.py       # Adam optimizer
│   ├── dynamics.py         # Glauber dynamics, exhaustive search
│   ├── regimes.py          # Four dynamical regimes from the paper
│   ├── replicas.py         # Multi-replica Glauber dynamics
│   ├── pruner.py           # GlauberPruner API (PyTorch-compatible)
│   ├── data.py             # Synthetic data generators
│   └── metrics.py          # Evaluation metrics
├── experiments/            # 25 experiments (numbered)
├── results/                # All CSV and JSON output files
├── tests/                  # 23 unit tests (all passing)
├── plots.html              # Interactive Chart.js visualizations (open in browser)
├── REPORT.md               # Full technical writeup
├── MEETING_REPORT.md       # Collaborator-ready summary with all findings
├── PIVOT.md                # Decision rationale and paper framing
└── perceptron_pruning_v5.1.r  # Original R reference implementation
```

---

## Reproducing Results

```bash
git clone https://github.com/corpetty/mozeika-pruning-empirics
cd mozeika-pruning-empirics
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Key experiments
python experiments/03_phase_diagram.py        # Phase transition in linear perceptron
python experiments/16_replica_rho_sweep.py    # Rényi replica sweep
python experiments/20_mlp_uwsh.py             # MLP — no phase transition
python experiments/25_sparsity_control_fix.py # Definitive baseline comparison
```

Results land in `results/`. Open `plots.html` in a browser for interactive charts of all experiments.

---

## Energy Function

The paper frames pruning as coupled Langevin dynamics over weights **w** and binary masks **h**:

```
E(w, h | D) = L(w ∘ h | D) + (η/2)||w||² + Σ V(hᵢ)
V(h) = α·h²(h-1)² + (ρ/2)·h
```

Four dynamical regimes emerge depending on the ratio of timescales τ_w / τ_h:
- **Equal timescales** → joint Boltzmann posterior
- **Fast learning** (τ_w ≪ τ_h) → Bayesian model selection via marginal likelihood; replica trick n = β_h/β_w
- **Fast pruning** (τ_h ≪ τ_w) → mask-averaged weight training
- **Low temperature** (β → ∞) → MAP alternating optimization (what Glauber dynamics implements)

---

## Original Reference

Mozeika, A. & Pizzoferrato, A. (2026). *Statistical mechanics of learning and pruning in neural networks.* Logos/Alan Turing Institute internal report.

---

## Related

- [Universal Weight Subspace Hypothesis](https://arxiv.org/abs/2512.05117) — Kaushik et al., JHU/Yuille lab
- [EigenLoRAx](https://arxiv.org/abs/2502.04700) — exploits shared spectral subspace
- Krzakala & Zdeborová (2019) — belief propagation direction for factor graph extension
