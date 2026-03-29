#!/usr/bin/env python3
"""
VGG16 3-way mask comparison on CIFAR-10.

Starts from the 90% sparsity checkpoint and prunes each method independently
to target sparsities [0.90, 0.95, 0.99] (with 90% as the starting point, so
we compare mask similarity at each additional pruning step).

Methods:
  1. Fisher/OBD (Mozeika-style)
  2. Magnitude (no rewind)
  3. Magnitude + rewind to 90% checkpoint weights

Outputs:
  - Per-method accuracy at each sparsity target
  - Jaccard similarity matrix for all 3 method pairs at each target
  - Plots: accuracy vs sparsity, Jaccard heatmaps per target level
"""

import copy
import os
import sys
import json
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Re-use model/data machinery from v4
sys.path.insert(0, os.path.dirname(__file__))
from vgg16_pruning_v4 import (
    MaskedConv2d, MaskedLinear, VGGPruningConfig,
    make_masked_vgg16, named_masked_modules, masked_modules,
    build_cifar10_loaders, evaluate, train_model, estimate_diag_fisher,
)

# ── Config ─────────────────────────────────────────────────────────────────────
CHECKPOINT_90   = "/home/petty/.openclaw/workspace-ai-research/vgg16_pruned_and_compressed.pt"
DATA_ROOT       = "/home/petty/.openclaw/workspace-ai-research/data"
RESULTS_DIR     = "/home/petty/pruning-research/results"
OUT_PREFIX      = "/home/petty/pruning-research/results/vgg16_mask_comparison"

# Sparsity targets to measure at (we start from 90% checkpoint)
SPARSITY_TARGETS = [0.95, 0.99]   # 90% is baseline; measure at 95% and 99%
PRUNE_FRAC_CAP  = 0.20            # max 20% of active weights per round
FINETUNE_EPOCHS = 2               # keep short — this is comparison not optimization
FISHER_BATCHES  = 10
LR              = 1e-4
SEED            = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ── Sparsity helpers ───────────────────────────────────────────────────────────
def get_global_sparsity(model: nn.Module) -> float:
    total = active = 0
    for _, m in named_masked_modules(model):
        total  += m.weight_mask.numel()
        active += m.weight_mask.sum().item()
        if hasattr(m, 'bias_mask') and m.bias_mask is not None:
            total  += m.bias_mask.numel()
            active += m.bias_mask.sum().item()
    return 1.0 - active / total if total > 0 else 0.0


def get_all_masks(model: nn.Module) -> Dict[str, np.ndarray]:
    masks = {}
    for name, m in named_masked_modules(model):
        masks[f"{name}.weight"] = m.weight_mask.data.cpu().numpy().copy()
        if hasattr(m, 'bias_mask') and m.bias_mask is not None:
            masks[f"{name}.bias"] = m.bias_mask.data.cpu().numpy().copy()
    return masks


def set_all_masks(model: nn.Module, masks: Dict[str, np.ndarray]) -> None:
    for name, m in named_masked_modules(model):
        wkey = f"{name}.weight"
        if wkey in masks:
            m.weight_mask.data.copy_(torch.from_numpy(masks[wkey]).to(m.weight_mask.device))
            m.weight.data.mul_(m.weight_mask)
        bkey = f"{name}.bias"
        if bkey in masks and hasattr(m, 'bias_mask') and m.bias_mask is not None:
            m.bias_mask.data.copy_(torch.from_numpy(masks[bkey]).to(m.bias_mask.device))
            if m.bias is not None:
                m.bias.data.mul_(m.bias_mask)


# ── Pruning ────────────────────────────────────────────────────────────────────
def prune_fisher(model: nn.Module, F_diag: Dict[str, torch.Tensor], target_sparsity: float) -> None:
    """OBD: S_i = 0.5 * F_ii * w_i^2, global threshold."""
    saliences = []
    layer_info = []
    for name, m in named_masked_modules(model):
        w    = m.weight.data.cpu().numpy().flatten()
        f    = F_diag.get(f"{name}.weight", torch.zeros_like(m.weight)).cpu().numpy().flatten()
        mask = m.weight_mask.data.cpu().numpy().flatten()
        sal  = 0.5 * f * w ** 2
        sal[mask == 0] = np.inf
        saliences.append(sal)
        layer_info.append(('weight', name, m, sal.shape))

        if hasattr(m, 'bias_mask') and m.bias_mask is not None and m.bias is not None:
            b    = m.bias.data.cpu().numpy().flatten()
            fb   = F_diag.get(f"{name}.bias", torch.zeros_like(m.bias)).cpu().numpy().flatten()
            bmask= m.bias_mask.data.cpu().numpy().flatten()
            bsal = 0.5 * fb * b ** 2
            bsal[bmask == 0] = np.inf
            saliences.append(bsal)
            layer_info.append(('bias', name, m, bsal.shape))

    combined  = np.concatenate(saliences)
    n_total   = combined.size
    n_target_active = int((1.0 - target_sparsity) * n_total)
    active_sal = combined[combined < np.inf]
    n_active  = len(active_sal)
    if n_active <= n_target_active:
        return
    n_prune = n_active - n_target_active
    # Cap per round
    n_prune = min(n_prune, int(PRUNE_FRAC_CAP * n_active))
    sorted_sal = np.sort(active_sal)
    threshold  = sorted_sal[n_prune - 1]

    offset = 0
    for kind, name, m, shape in layer_info:
        sz   = int(np.prod(shape))
        sal_chunk = saliences[layer_info.index((kind, name, m, shape))][:]
        if kind == 'weight':
            w    = m.weight.data.cpu().numpy()
            f    = F_diag.get(f"{name}.weight", torch.zeros_like(m.weight)).cpu().numpy()
            mask = m.weight_mask.data.cpu().numpy()
            sal2d = (0.5 * f * w**2)
            new_mask = np.where((sal2d <= threshold) & (mask == 1), 0.0, mask)
            m.weight_mask.data.copy_(torch.from_numpy(new_mask).to(m.weight_mask.device))
            m.weight.data.mul_(m.weight_mask)
        else:
            b    = m.bias.data.cpu().numpy()
            fb   = F_diag.get(f"{name}.bias", torch.zeros_like(m.bias)).cpu().numpy()
            bmask= m.bias_mask.data.cpu().numpy()
            bsal = 0.5 * fb * b**2
            new_bmask = np.where((bsal <= threshold) & (bmask == 1), 0.0, bmask)
            m.bias_mask.data.copy_(torch.from_numpy(new_bmask).to(m.bias_mask.device))
            m.bias.data.mul_(m.bias_mask)


def prune_magnitude(model: nn.Module, target_sparsity: float) -> None:
    """Global magnitude pruning."""
    magnitudes = []
    layer_info = []
    for name, m in named_masked_modules(model):
        w    = m.weight.data.cpu().numpy().flatten()
        mask = m.weight_mask.data.cpu().numpy().flatten()
        mag  = np.abs(w)
        mag[mask == 0] = np.inf
        magnitudes.append(mag)
        layer_info.append(('weight', name, m))

        if hasattr(m, 'bias_mask') and m.bias_mask is not None and m.bias is not None:
            b    = m.bias.data.cpu().numpy().flatten()
            bmask= m.bias_mask.data.cpu().numpy().flatten()
            bmag = np.abs(b)
            bmag[bmask == 0] = np.inf
            magnitudes.append(bmag)
            layer_info.append(('bias', name, m))

    combined  = np.concatenate(magnitudes)
    n_total   = combined.size
    n_target_active = int((1.0 - target_sparsity) * n_total)
    active_mag = combined[combined < np.inf]
    n_active  = len(active_mag)
    if n_active <= n_target_active:
        return
    n_prune = n_active - n_target_active
    n_prune = min(n_prune, int(PRUNE_FRAC_CAP * n_active))
    threshold = np.sort(active_mag)[n_prune - 1]

    for kind, name, m in layer_info:
        if kind == 'weight':
            w    = m.weight.data.cpu().numpy()
            mask = m.weight_mask.data.cpu().numpy()
            new_mask = np.where((np.abs(w) <= threshold) & (mask == 1), 0.0, mask)
            m.weight_mask.data.copy_(torch.from_numpy(new_mask).to(m.weight_mask.device))
            m.weight.data.mul_(m.weight_mask)
        else:
            b    = m.bias.data.cpu().numpy()
            bmask= m.bias_mask.data.cpu().numpy()
            new_bmask = np.where((np.abs(b) <= threshold) & (bmask == 1), 0.0, bmask)
            m.bias_mask.data.copy_(torch.from_numpy(new_bmask).to(m.bias_mask.device))
            m.bias.data.mul_(m.bias_mask)


# ── Jaccard ────────────────────────────────────────────────────────────────────
def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten() > 0
    b = b.flatten() > 0
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union > 0 else 1.0


def cross_jaccard_masks(masks_a: Dict, masks_b: Dict) -> Dict:
    all_a, all_b, per_key = [], [], {}
    for k in masks_a:
        if k in masks_b:
            j = jaccard(masks_a[k], masks_b[k])
            per_key[k] = j
            all_a.append(masks_a[k].flatten())
            all_b.append(masks_b[k].flatten())
    return {'global': jaccard(np.concatenate(all_a), np.concatenate(all_b)), **per_key}


# ── Load checkpoint ────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path: str, cfg: VGGPruningConfig) -> nn.Module:
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.use_pretrained = False
    model = make_masked_vgg16(cfg_copy).to(cfg.device)
    ckpt  = torch.load(ckpt_path, map_location=cfg.device)
    key   = 'masked_state_dict' if 'masked_state_dict' in ckpt else 'model_state_dict'
    if key not in ckpt:
        key = list(ckpt.keys())[0]
    model.load_state_dict(ckpt[key], strict=False)
    return model


# ── Single method run ──────────────────────────────────────────────────────────
def run_method(name: str, base_model: nn.Module, base_weights: Dict[str, torch.Tensor],
               train_loader: DataLoader, test_loader: DataLoader, cfg: VGGPruningConfig):
    print(f"\n{'='*60}\nMethod: {name.upper()}\n{'='*60}")
    model = copy.deepcopy(base_model).to(cfg.device)

    snapshots = {}  # sparsity_target -> masks
    acc_log   = {}  # sparsity_target -> accuracy
    sp_log    = {}

    current_target_idx = 0
    targets = list(SPARSITY_TARGETS)
    rnd = 0

    while current_target_idx < len(targets):
        target = targets[current_target_idx]
        sp = get_global_sparsity(model)
        print(f"  Round {rnd+1} | current sparsity={sp:.4f} | target={target:.2f}")

        # Prune one step toward target
        if name == 'fisher':
            F_diag = estimate_diag_fisher_named(model, train_loader, cfg)
            prune_fisher(model, F_diag, target)
        elif name == 'magnitude':
            prune_magnitude(model, target)
        elif name == 'magnitude_rewind':
            prune_magnitude(model, target)
            # Rewind weights to base (90% checkpoint), keep new masks
            new_masks = get_all_masks(model)
            # Reset weights to base_weights, re-apply mask
            for n, m in named_masked_modules(model):
                wkey = f"{n}.weight"
                if wkey in base_weights:
                    with torch.no_grad():
                        m.weight.data.copy_(base_weights[wkey].to(m.weight.device))
                bkey = f"{n}.bias"
                if bkey in base_weights and m.bias is not None:
                    with torch.no_grad():
                        m.bias.data.copy_(base_weights[bkey].to(m.bias.device))
            set_all_masks(model, new_masks)

        # Fine-tune
        train_model(model, train_loader, cfg, epochs=FINETUNE_EPOCHS)

        sp_after = get_global_sparsity(model)
        _, acc = evaluate(model, test_loader, cfg.device)
        print(f"  → sparsity={sp_after:.4f} acc={acc:.4f}")

        rnd += 1

        if sp_after >= target - 0.005 or rnd > 40:
            # Snapshot masks at this target
            snapshots[target] = get_all_masks(model)
            acc_log[target] = acc
            sp_log[target]  = sp_after
            print(f"  ✓ Snapshot at target={target:.2f}: sparsity={sp_after:.4f} acc={acc:.4f}")
            current_target_idx += 1

    # Free GPU memory
    model.cpu()
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return snapshots, acc_log, sp_log


def estimate_diag_fisher_named(model: nn.Module, loader: DataLoader, cfg: VGGPruningConfig) -> Dict[str, torch.Tensor]:
    """Returns F[name.weight] and F[name.bias] tensors."""
    F_diag: Dict[str, torch.Tensor] = {}
    for name, m in named_masked_modules(model):
        F_diag[f"{name}.weight"] = torch.zeros_like(m.weight)
        if hasattr(m, 'bias_mask') and m.bias_mask is not None and m.bias is not None:
            F_diag[f"{name}.bias"] = torch.zeros_like(m.bias)

    model.eval()
    count = 0
    for x, y in loader:
        if count >= FISHER_BATCHES:
            break
        x, y = x.to(cfg.device), y.to(cfg.device)
        model.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        for name, m in named_masked_modules(model):
            if m.weight.grad is not None:
                F_diag[f"{name}.weight"] += m.weight.grad.data ** 2
            if hasattr(m, 'bias_mask') and m.bias_mask is not None and m.bias is not None:
                if m.bias.grad is not None:
                    F_diag[f"{name}.bias"] += m.bias.grad.data ** 2
        count += 1

    for k in F_diag:
        F_diag[k] /= max(count, 1)
    return F_diag


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cfg = VGGPruningConfig(
        device=DEVICE,
        seed=SEED,
        batch_size=32,
        lr=LR,
        finetune_epochs=FINETUNE_EPOCHS,
        fisher_batches=FISHER_BATCHES,
        data_root=DATA_ROOT,
        use_pretrained=False,
        image_size=224,
        num_workers=4,
    )

    print(f"Device: {cfg.device}")
    print(f"Loading 90% checkpoint: {CHECKPOINT_90}")
    base_model = load_checkpoint(CHECKPOINT_90, cfg)
    sp0 = get_global_sparsity(base_model)
    train_loader, test_loader = build_cifar10_loaders(cfg)
    _, acc0 = evaluate(base_model, test_loader, cfg.device)
    print(f"Baseline (90% checkpoint): sparsity={sp0:.4f} acc={acc0:.4f}")

    # Capture base weights (for rewind method)
    base_weights = {}
    for name, m in named_masked_modules(base_model):
        base_weights[f"{name}.weight"] = m.weight.data.cpu().clone()
        if m.bias is not None:
            base_weights[f"{name}.bias"] = m.bias.data.cpu().clone()

    # Run all three methods (with per-method checkpoint so we can resume)
    results = {}
    PARTIAL_CACHE = f"{OUT_PREFIX}_partial.pt"
    if os.path.exists(PARTIAL_CACHE):
        print(f"Loading partial results from {PARTIAL_CACHE}")
        results = torch.load(PARTIAL_CACHE, map_location='cpu')
        print(f"  Already done: {list(results.keys())}")

    for method in ['fisher', 'magnitude', 'magnitude_rewind']:
        if method in results:
            print(f"Skipping {method} (already done)")
            continue
        snaps, accs, sps = run_method(method, base_model, base_weights,
                                      train_loader, test_loader, cfg)
        results[method] = {'snapshots': snaps, 'accs': accs, 'sps': sps}
        torch.save(results, PARTIAL_CACHE)
        print(f"  Saved partial results after {method}")
        # Explicitly free GPU memory between methods
        torch.cuda.empty_cache()
        import gc; gc.collect()

    # Compute cross-method Jaccard at each target
    methods = ['fisher', 'magnitude', 'magnitude_rewind']
    pairs = [('fisher', 'magnitude'), ('fisher', 'magnitude_rewind'), ('magnitude', 'magnitude_rewind')]
    pair_labels = ['Fisher vs Mag', 'Fisher vs Mag+Rew', 'Mag vs Mag+Rew']

    all_jaccard = {}
    for target in SPARSITY_TARGETS:
        all_jaccard[target] = {}
        print(f"\n{'='*50}")
        print(f"MASK OVERLAP at target sparsity {target:.0%}")
        print(f"{'Pair':<30} {'global':>8}")
        print("-"*40)
        for (m1, m2), label in zip(pairs, pair_labels):
            if target in results[m1]['snapshots'] and target in results[m2]['snapshots']:
                j = cross_jaccard_masks(results[m1]['snapshots'][target],
                                        results[m2]['snapshots'][target])
                all_jaccard[target][(m1, m2)] = j
                print(f"  {label:<28} {j['global']:>8.4f}  "
                      f"(acc: {results[m1]['accs'].get(target,'?'):.4f} / "
                      f"{results[m2]['accs'].get(target,'?'):.4f})")

    # Save JSON results
    json_out = {}
    for target in SPARSITY_TARGETS:
        json_out[str(target)] = {
            'accuracy': {m: results[m]['accs'].get(target) for m in methods},
            'sparsity': {m: results[m]['sps'].get(target) for m in methods},
            'jaccard': {}
        }
        for (m1, m2) in pairs:
            key = f"{m1}_vs_{m2}"
            if (m1, m2) in all_jaccard.get(target, {}):
                j = all_jaccard[target][(m1, m2)]
                json_out[str(target)]['jaccard'][key] = j.get('global')
    with open(f"{OUT_PREFIX}_results.json", 'w') as f:
        json.dump({'baseline': {'sparsity': sp0, 'accuracy': acc0}, **json_out}, f, indent=2)
    print(f"\nResults saved: {OUT_PREFIX}_results.json")

    # ── Plots ──────────────────────────────────────────────────────────────────
    COLORS = {'fisher': '#2196F3', 'magnitude': '#F44336', 'magnitude_rewind': '#4CAF50'}
    LABELS = {'fisher': 'Fisher/OBD', 'magnitude': 'Magnitude', 'magnitude_rewind': 'Mag+Rewind'}

    # 1. Accuracy at each sparsity target
    fig, axes = plt.subplots(1, len(SPARSITY_TARGETS), figsize=(5 * len(SPARSITY_TARGETS) + 2, 5),
                              facecolor='white', sharey=False)
    if len(SPARSITY_TARGETS) == 1:
        axes = [axes]

    for ti, target in enumerate(SPARSITY_TARGETS):
        ax = axes[ti]
        method_names = list(LABELS.keys())
        accs_here = [results[m]['accs'].get(target, 0) for m in method_names]
        bars = ax.bar(range(len(method_names)), accs_here,
                      color=[COLORS[m] for m in method_names], alpha=0.85)
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels([LABELS[m] for m in method_names], rotation=15, ha='right')
        ax.set_ylim(max(0, min(accs_here) - 0.05), min(1.0, max(accs_here) + 0.05))
        ax.set_title(f'Accuracy @ {target:.0%} Sparsity')
        ax.set_ylabel('Test Accuracy')
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, acc in zip(bars, accs_here):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{acc:.3f}', ha='center', fontsize=9)

    fig.suptitle('VGG16 CIFAR-10: Accuracy by Method', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT_PREFIX}_accuracy.png", dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Accuracy plot saved.")

    # 2. Jaccard heatmaps
    for target in SPARSITY_TARGETS:
        fig2, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        jac_matrix = np.ones((3, 3))
        for i, m1 in enumerate(methods):
            for j2, m2 in enumerate(methods):
                if m1 == m2:
                    jac_matrix[i, j2] = 1.0
                else:
                    key = (m1, m2) if (m1, m2) in all_jaccard.get(target, {}) else (m2, m1)
                    if key in all_jaccard.get(target, {}):
                        jac_matrix[i, j2] = all_jaccard[target][key]['global']

        im = ax.imshow(jac_matrix, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        short = ['Fisher', 'Magnitude', 'Mag+\nRewind']
        ax.set_xticklabels(short)
        ax.set_yticklabels(short)
        for i in range(3):
            for j2 in range(3):
                ax.text(j2, i, f'{jac_matrix[i,j2]:.3f}', ha='center', va='center',
                        fontsize=11, color='white' if jac_matrix[i,j2] > 0.6 else 'black')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Mask Jaccard @ {target:.0%} Sparsity\n(VGG16 CIFAR-10)')
        ax.set_facecolor('white')
        plt.tight_layout()
        plt.savefig(f"{OUT_PREFIX}_jaccard_{int(target*100)}pct.png",
                    dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Jaccard heatmap saved for {target:.0%}.")

    plt.close('all')
    print(f"\nAll done. Results in {RESULTS_DIR}")


if __name__ == '__main__':
    main()
