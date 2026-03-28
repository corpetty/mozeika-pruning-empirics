# mozeika-pruning-empirics

Empirical evaluation of neural network pruning and compression methods, built over ~4 months across three interconnected research threads.

**Hardware:** 2× NVIDIA RTX 3090 (24GB each), 504GB RAM, machine "bugger"
**Python env:** `/home/petty/torch-env` (PyTorch 2.x + CUDA)

---

## Repository Structure

```
├── experiments/          # Mozeika statistical mechanics pruning (Exps 1–29)
├── pruning_core/         # Shared Python library for pruning experiments
├── results/              # CSVs and output from Mozeika experiments
├── kv-subspace/          # KV cache compression via PCA subspace + PolarQuant (Exps 1–12)
│   ├── experiments/      # 12 experiment scripts
│   ├── results/          # CSVs, 12 per-experiment reports, SUMMARY.md, 6 figures
│   └── scripts/          # Plotting utilities
├── vgg16-fisher/         # VGG16 Fisher information pruning on CIFAR-10
│   ├── vgg16_pruning.py  # Main pruning script
│   └── VGG16_RESULTS.md  # Results + engineering notes
├── papers/               # Paper summaries and reading notes
├── notes/                # Active research threads and scratch notes
├── EXPLAINER.md          # Plain-language guide for non-physicists
├── RESEARCH_PLAN.md      # Research arc and open questions
├── GLAUBER_RESULTS.md    # Glauber dynamics results (LeNet-300-100 on MNIST)
├── REPORT.md             # Main Mozeika evaluation report
├── MEETING_REPORT.md     # Summary for collaborators
└── plots.html            # Interactive plots
```

---

## Research Threads

### Thread 1: Mozeika Statistical Mechanics Pruning (`experiments/`)

**Paper:** Mozeika & Pizzoferrato (2026) — statistical mechanics framework for neural network pruning, using replica method to predict a phase transition at critical sparsity ρ_c.

**25 experiments, key findings:**
- Phase transition is real and observable in linear perceptrons (ρ_c ≈ 0.0001 confirmed)
- Does **not** generalize to MLPs or CNNs
- Mozeika's ρ_c formula off by 100–20,000× for real architectures
- Root cause: mean-field artifact that breaks down for non-linear, deep networks
- L1 regularization beats Mozeika at every sparsity level
- **Verdict:** Negative result. The physics intuition is right; the theoretical predictions are wrong for practical networks.

See `REPORT.md`, `MEETING_REPORT.md`, `PIVOT.md`.

### Thread 2: Glauber Dynamics Pruning (`experiments/`, `lenet300_pruning_finite_temp.py`)

**Inspired by:** Mozeika's statistical mechanics framework — using thermal annealing to explore weight masks.

**LeNet-300-100 on MNIST, Runs 1–10, key findings:**
- 99.1% sparsity achievable at 97.2–97.5% accuracy
- Glauber anneal beats iterative magnitude pruning by +1.08% at 99% sparsity
- Three-phase structure: rapid pruning → restructuring plateau → thermal collapse
- Collapse is thermodynamic (T→0), not equilibration-limited — more sweeps don't help

See `GLAUBER_RESULTS.md`.

### Thread 3: KV Cache Compression (`kv-subspace/`)

**Method:** PCA subspace projection + PolarQuant quantization for transformer KV caches.

**Qwen3-14B-AWQ baseline → Mistral-7B, Phi-4, Qwen3-1.7B, Qwen3-32B (12 experiments), key findings:**
- 4× KV cache compression with <15% PPL degradation is achievable
- **Critical insight (Exp 9):** Truncation error dominates over quantization noise — k=64/16-bit is 2.48× PPL degradation; k=128/4-bit is 1.05×. Use bigger k, not more bits.
- K vectors more compressible than V (mean effective rank ~30 vs ~54)
- Compression tolerance scales with model size; also architecture-dependent (Mistral/Phi3 more tolerant than Qwen3)
- Hardware overhead ~1.7× latency; fixable with fused CUDA kernel

Recommended configs for Qwen3-14B-AWQ:
- `k=128/4-bit` → 4.00× compression, 1.05× PPL (safe default)
- `k=112/4-bit` → 4.27× compression, 1.14× PPL (max within 20% threshold)

See `kv-subspace/results/SUMMARY.md`, `kv-subspace/results/REPORT-*.md`.

### Thread 4: VGG16 Fisher Pruning (`vgg16-fisher/`)

**Method:** Iterative Fisher information magnitude pruning on VGG16 (ImageNet pretrained) for CIFAR-10.

**Key result:** 90% of weights removed, accuracy improves from 89.94% → **93.06% (+3.12%)**.
- Conv filters nearly all useful (502/512 survive)
- FC layers massively overparameterized for CIFAR-10 (~40×)
- Fisher saliency with 3 mini-batches sufficient for reliable ranking

See `vgg16-fisher/VGG16_RESULTS.md`.

---

## Cross-Thread Themes

1. **Principled saliency beats random pruning** — Fisher, Glauber, and PCA subspace all outperform magnitude/random baselines when the method correctly identifies "what matters"
2. **Overparameterization is the real variable** — VGG16 gains accuracy when pruned (massively overparameterized); LeNet loses a small amount (closer to right-sized); KV caches lose tolerably (redundant dimensions)
3. **Theory underpredicts practical thresholds** — Mozeika's ρ_c is off by orders of magnitude; KV truncation tolerance depends on model size in ways simple theory doesn't capture
4. **The compression budget is in quantization, not dimensionality reduction** — KV subspace lesson: don't truncate aggressively, quantize aggressively

---

## Quick Start

```bash
# Mozeika/Glauber experiments
cd ~/pruning-research
source /home/petty/torch-env/bin/activate
python experiments/01_perceptron_glauber.py
python lenet300_pruning_finite_temp.py

# KV-subspace experiments
cd kv-subspace
python experiments/perplexity_eval.py

# VGG16 Fisher pruning
cd vgg16-fisher
python vgg16_pruning.py
```

---

## Next Directions

- **Belief Propagation pruning** (Krzakala/Zdeborová 2019) — handles non-linear networks more rigorously than Mozeika's mean-field approach
- **Fused CUDA kernel** for KV subspace projection to close the 1.7× latency gap
- **Glauber on transformers** — can the three-phase dynamics be exploited for structured pruning of attention heads?
- **Cross-thread connection** — KV cache compression is essentially subspace pruning of activation tensors; Glauber dynamics could explore KV mask space

---

## Papers

See `papers/` for summaries:
- `mozeika-pizzoferrato-2026-pruning.md` — the paper being evaluated
- `mozeika-pizzoferrato-explainer.md` — plain-language walkthrough
- `universal-weight-subspace-hypothesis.md` — UWSH background

See also `EXPLAINER.md` for an accessible overview of the entire research program.
