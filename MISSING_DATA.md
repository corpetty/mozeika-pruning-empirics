# Missing Data — What's Not in This Repo

Several large files are excluded from git (`.gitignore`) because they exceed GitHub's 100MB limit or are auto-generated. This document explains what's missing, where it lives on the original machine, and how to reproduce each item.

---

## 1. VGG16 Compressed Model Checkpoint

**File:** `vgg16_pruned_and_compressed.pt` (~1.4GB)
**What it is:** The final compressed VGG16 model after round 11 of Fisher pruning (90.1% sparsity, 93.06% accuracy on CIFAR-10).

**How to reproduce:**
```bash
cd vgg16-fisher
# Requires VGG16 pretrained weights (auto-downloaded by torchvision on first run)
# Requires CIFAR-10 dataset (auto-downloaded on first run)
python vgg16_pruning.py
```
Runtime: ~2–3 hours on a single GPU. GPU with ≥12GB VRAM recommended (script defaults to `cuda:0`). See `VGG16_RESULTS.md` for full config used in the original run (`fisher_batches=3`, `max_pruning_rounds=25`, `prune_fraction_cap=0.15`, `device=cuda:1`).

---

## 2. KV-Subspace NumPy Analysis Files

**Files:** `kv-subspace/results/*.npz` (several files, ~750MB total)
**What they are:** Raw KV cache activation tensors captured from Qwen3-14B-AWQ during calibration. Used as input to all 12 KV compression experiments.

**Key files:**
- `kv-subspace/results/kvs.npz` — primary calibration data (Qwen3-14B-AWQ, WikiText-2 domain)
- `kv-subspace/results/kvs_domain2.npz` — cross-domain calibration (code domain)
- `kv-subspace/results/analysis.npz` — intermediate PCA analysis cache

**How to reproduce:**
```bash
cd kv-subspace
# Requires Qwen3-14B-AWQ loaded in vLLM or via transformers
# Edit collect.py to point at your model endpoint
python collect.py          # captures KV tensors → kvs.npz
python analyze.py          # runs PCA analysis → analysis.npz
```
Model used: `Qwen/Qwen3-14B-AWQ` (~8.5GB, AWQ 4-bit). See `collect.py` for the exact calibration corpus (WikiText-2, 512 tokens, 100 sequences). Cross-domain file uses the CodeSearchNet corpus.

---

## 3. CIFAR-10 Dataset

**Directory:** `data/` and `vgg16-fisher/data/` (~163MB)
**What it is:** CIFAR-10 image classification dataset (Python pickle format).

**How to reproduce:** Auto-downloaded by the scripts on first run via `torchvision.datasets.CIFAR10(download=True)`. No manual steps needed.

---

## 4. VGG16 Pretrained Weights

**File:** `~/.cache/torch/hub/checkpoints/vgg16-397923af.pth` (~528MB, outside repo)
**What it is:** ImageNet-pretrained VGG16 weights from PyTorch model zoo.

**How to reproduce:** Auto-downloaded by torchvision on first run. Or manually:
```bash
wget https://download.pytorch.org/models/vgg16-397923af.pth \
     -P ~/.cache/torch/hub/checkpoints/
```

---

## Summary Table

| Missing file/dir | Size | How to get it |
|---|---|---|
| `vgg16_pruned_and_compressed.pt` | ~1.4GB | Run `vgg16-fisher/vgg16_pruning.py` |
| `kv-subspace/results/kvs.npz` | ~500MB | Run `kv-subspace/collect.py` with Qwen3-14B-AWQ |
| `kv-subspace/results/kvs_domain2.npz` | ~150MB | Run `kv-subspace/collect.py` (code domain config) |
| `kv-subspace/results/analysis.npz` | ~100MB | Run `kv-subspace/analyze.py` after collect |
| `data/` (CIFAR-10) | ~163MB | Auto-downloaded on first run |
| `~/.cache/torch/.../vgg16-*.pth` | ~528MB | Auto-downloaded or via wget (see above) |
