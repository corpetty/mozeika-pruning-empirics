# KV Subspace Compression

**Hypothesis:** KV cache vectors in transformer attention heads live in low-dimensional
subspaces (per the Universal Weight Subspace Hypothesis). Compressing in the principal
subspace should achieve better quality-per-bit than full-dimensional compression.

**Experiment:**
1. Extract K/V vectors from each layer/head of a model on a long-context dataset
2. Measure effective rank (PCA spectral decay) per layer/head
3. Compare attention score distortion: full-dim PolarQuant vs subspace PolarQuant
   at matched bits/vector budgets
4. Plot: effective rank vs layer depth, and bits-to-distortion curves per layer

**Stack:** Qwen3-14B-AWQ (already local), LongBench/PG-19 for text, custom KV hooks.

## Repo structure

```
collect.py        — hook into model, collect K/V vectors per layer/head
analyze.py        — PCA, effective rank, spectral decay plots
compress.py       — PolarQuant + QJL implementation, distortion measurement
experiment.py     — full pipeline: collect → compress → report
results/          — CSVs and plots
report.html       — interactive Chart.js report
```

## Quick start

```bash
cd /home/petty/kv-subspace
source venv/bin/activate
python experiment.py --model Qwen/Qwen3-14B-AWQ --n-tokens 4096 --output results/
```
