#!/usr/bin/env python3
"""
Experiment 30: Cross-method mask comparison on LeNet-300-100 + MNIST

Three methods all starting from the SAME pretrained checkpoint:
  1. Fisher/OBD iterative pruning (Mozeika-style)
  2. Magnitude (no rewind) iterative pruning
  3. Magnitude + rewind (reset to pretrained weights after each prune)

Target: 99% sparsity, iterative with fine-tuning each round.
Metrics: final accuracy, Jaccard similarity between all 3 pairs (global + per-layer),
         and a heatmap of which specific weights each method kills.
"""

import os
import sys
import csv
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from copy import deepcopy

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 128
PRETRAIN_EPOCHS = 10
FINETUNE_EPOCHS = 5
LR            = 1e-3
PRUNE_FRAC    = 0.20   # prune 20% of remaining active weights per round
TARGET        = 0.99
FISHER_BATCHES = 100
SEED          = 42
RESULTS_DIR   = "/home/petty/pruning-research/results"
OUT_PREFIX    = "/home/petty/pruning-research/results/lenet_mask_comparison"
DATA_DIR      = "/home/petty/pruning-research/data"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Model ──────────────────────────────────────────────────────────────────────
class MaskedLinear(nn.Linear):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__(in_f, out_f, bias)
        self.register_buffer('weight_mask', torch.ones_like(self.weight))

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.weight_mask, self.bias)

    def apply_mask(self, mask_np):
        self.weight_mask.data.copy_(torch.from_numpy(mask_np).to(self.weight_mask.device))
        with torch.no_grad():
            self.weight.data *= self.weight_mask

    def zero_masked_grads(self):
        if self.weight.grad is not None:
            self.weight.grad.data *= self.weight_mask


class LeNet300100(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = MaskedLinear(784, 300)
        self.fc2 = MaskedLinear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def masked_layers(self):
        return [('fc1', self.fc1), ('fc2', self.fc2)]

    def get_masks(self):
        return {n: l.weight_mask.data.cpu().numpy().copy()
                for n, l in self.masked_layers()}

    def set_masks(self, masks):
        for n, l in self.masked_layers():
            l.apply_mask(masks[n])

    def get_sparsity(self):
        total = active = 0
        for _, l in self.masked_layers():
            total  += l.weight_mask.numel()
            active += l.weight_mask.sum().item()
        return 1.0 - active / total

    def get_layer_sparsity(self):
        out = {}
        for n, l in self.masked_layers():
            out[n] = 1.0 - l.weight_mask.float().mean().item()
        return out


# ── Data ───────────────────────────────────────────────────────────────────────
def get_loaders():
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    os.makedirs(DATA_DIR, exist_ok=True)
    train = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
    test  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2),
            DataLoader(test,  batch_size=256,        shuffle=False, num_workers=2))


# ── Train / eval ───────────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, criterion):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        for _, l in model.masked_layers():
            l.zero_masked_grads()
        opt.step()
        # Re-zero pruned weights
        with torch.no_grad():
            for _, l in model.masked_layers():
                l.weight.data *= l.weight_mask


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / total


# ── Fisher diagonal ────────────────────────────────────────────────────────────
def estimate_fisher(model, loader, n_batches):
    F = {n: torch.zeros_like(l.weight) for n, l in model.masked_layers()}
    model.eval()
    count = 0
    for x, y in loader:
        if count >= n_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        model.zero_grad()
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        for n, l in model.masked_layers():
            if l.weight.grad is not None:
                F[n] += l.weight.grad.data ** 2
        count += 1
    for n in F:
        F[n] /= count
    return F


# ── Pruning methods ────────────────────────────────────────────────────────────
def prune_fisher(model, F, target_sparsity):
    """OBD saliency: S_i = 0.5 * F_ii * w_i^2"""
    layers = model.masked_layers()
    saliences = []
    for n, l in layers:
        w    = l.weight.data.cpu().numpy().flatten()
        f    = F[n].cpu().numpy().flatten()
        mask = l.weight_mask.data.cpu().numpy().flatten()
        sal  = 0.5 * f * w ** 2
        sal[mask == 0] = np.inf
        saliences.append(sal)
    combined = np.concatenate(saliences)
    n_active = (combined < np.inf).sum()
    n_target = int((1.0 - target_sparsity) * (combined.size))
    if n_target >= n_active:
        return
    # global threshold
    active_sal = combined[combined < np.inf]
    if len(active_sal) == 0:
        return
    n_prune = max(1, n_active - n_target)
    threshold = np.partition(active_sal, n_prune - 1)[n_prune - 1]
    for n, l in layers:
        w    = l.weight.data.cpu().numpy()
        f    = F[n].cpu().numpy()
        mask = l.weight_mask.data.cpu().numpy()
        sal  = 0.5 * f * w ** 2
        new_mask = np.where((sal <= threshold) & (mask == 1), 0.0, mask)
        l.apply_mask(new_mask)


def prune_magnitude(model, target_sparsity):
    """Global magnitude pruning."""
    layers = model.masked_layers()
    magnitudes = []
    for n, l in layers:
        w    = l.weight.data.cpu().numpy().flatten()
        mask = l.weight_mask.data.cpu().numpy().flatten()
        mag  = np.abs(w)
        mag[mask == 0] = np.inf
        magnitudes.append(mag)
    combined = np.concatenate(magnitudes)
    n_active = (combined < np.inf).sum()
    n_target = int((1.0 - target_sparsity) * combined.size)
    if n_target >= n_active:
        return
    active_mag = combined[combined < np.inf]
    n_prune = max(1, n_active - n_target)
    threshold = np.partition(active_mag, n_prune - 1)[n_prune - 1]
    for n, l in layers:
        w    = l.weight.data.cpu().numpy()
        mask = l.weight_mask.data.cpu().numpy()
        new_mask = np.where((np.abs(w) <= threshold) & (mask == 1), 0.0, mask)
        l.apply_mask(new_mask)


# ── Jaccard ────────────────────────────────────────────────────────────────────
def jaccard(mask_a, mask_b):
    a = mask_a.flatten() > 0
    b = mask_b.flatten() > 0
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / union if union > 0 else 1.0


def cross_jaccard(masks_a, masks_b):
    """Per-layer and global Jaccard between two mask dicts."""
    results = {}
    all_a, all_b = [], []
    for layer in masks_a:
        j = jaccard(masks_a[layer], masks_b[layer])
        results[layer] = j
        all_a.append(masks_a[layer].flatten())
        all_b.append(masks_b[layer].flatten())
    results['global'] = jaccard(np.concatenate(all_a), np.concatenate(all_b))
    return results


# ── Single method run ──────────────────────────────────────────────────────────
def run_method(name, pretrained_state, pretrained_weights, train_loader, test_loader):
    print(f"\n{'='*60}\nMethod: {name}\n{'='*60}")
    model = LeNet300100().to(DEVICE)
    model.load_state_dict(pretrained_state)
    criterion = nn.CrossEntropyLoss()

    round_num = 0
    log = []

    while model.get_sparsity() < TARGET:
        round_num += 1
        current_sp = model.get_sparsity()

        if name == 'fisher':
            F = estimate_fisher(model, train_loader, FISHER_BATCHES)
            target_sp = min(TARGET, current_sp + (1.0 - current_sp) * PRUNE_FRAC)
            prune_fisher(model, F, target_sp)
        elif name == 'magnitude':
            target_sp = min(TARGET, current_sp + (1.0 - current_sp) * PRUNE_FRAC)
            prune_magnitude(model, target_sp)
        elif name == 'magnitude_rewind':
            target_sp = min(TARGET, current_sp + (1.0 - current_sp) * PRUNE_FRAC)
            # Save current mask, reset to pretrained weights, apply mask
            current_masks = model.get_masks()
            prune_magnitude(model, target_sp)
            new_masks = model.get_masks()
            # Rewind weights to pretrained, re-apply new mask
            for n, l in model.masked_layers():
                with torch.no_grad():
                    l.weight.data.copy_(pretrained_weights[n])
                l.apply_mask(new_masks[n])

        # Fine-tune
        opt = optim.Adam(model.parameters(), lr=LR)
        for ep in range(FINETUNE_EPOCHS):
            train_epoch(model, train_loader, opt, criterion)

        sp  = model.get_sparsity()
        acc = evaluate(model, test_loader)
        lsp = model.get_layer_sparsity()
        print(f"  Round {round_num:2d} | sparsity={sp:.4f} | acc={acc:.4f} | "
              f"fc1={lsp['fc1']:.3f} fc2={lsp['fc2']:.3f}")
        log.append({'round': round_num, 'sparsity': sp, 'acc': acc, **lsp})

        if round_num > 60:
            print("  Max rounds reached, stopping.")
            break

    final_acc  = evaluate(model, test_loader)
    final_sp   = model.get_sparsity()
    final_masks = model.get_masks()
    print(f"\n  FINAL: sparsity={final_sp:.4f} acc={final_acc:.4f}")
    return final_masks, final_acc, final_sp, log


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_loader, test_loader = get_loaders()

    # ── Pretrain shared model ──
    print("Pretraining shared model...")
    model = LeNet300100().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)
    for ep in range(PRETRAIN_EPOCHS):
        train_epoch(model, train_loader, opt, criterion)
        acc = evaluate(model, test_loader)
        print(f"  Epoch {ep+1}/{PRETRAIN_EPOCHS} — acc={acc:.4f}")

    pretrained_state = deepcopy(model.state_dict())
    # Capture raw pretrained weights for rewind (before any masking)
    pretrained_weights = {n: l.weight.data.cpu().clone()
                          for n, l in model.masked_layers()}

    # ── Run all three methods ──
    masks_fisher,   acc_fisher,   sp_fisher,   log_fisher   = run_method('fisher',           pretrained_state, pretrained_weights, train_loader, test_loader)
    masks_mag,      acc_mag,      sp_mag,      log_mag      = run_method('magnitude',         pretrained_state, pretrained_weights, train_loader, test_loader)
    masks_mag_rew,  acc_mag_rew,  sp_mag_rew,  log_mag_rew  = run_method('magnitude_rewind',  pretrained_state, pretrained_weights, train_loader, test_loader)

    # ── Cross-method Jaccard ──
    j_f_m   = cross_jaccard(masks_fisher, masks_mag)
    j_f_mr  = cross_jaccard(masks_fisher, masks_mag_rew)
    j_m_mr  = cross_jaccard(masks_mag,    masks_mag_rew)

    print("\n" + "="*60)
    print("MASK OVERLAP (Jaccard similarity — 1.0 = identical)")
    print(f"{'Pair':<30} {'fc1':>8} {'fc2':>8} {'global':>8}")
    print("-"*56)
    print(f"{'Fisher vs Magnitude':<30} {j_f_m['fc1']:>8.4f} {j_f_m['fc2']:>8.4f} {j_f_m['global']:>8.4f}")
    print(f"{'Fisher vs Mag+Rewind':<30} {j_f_mr['fc1']:>8.4f} {j_f_mr['fc2']:>8.4f} {j_f_mr['global']:>8.4f}")
    print(f"{'Magnitude vs Mag+Rewind':<30} {j_m_mr['fc1']:>8.4f} {j_m_mr['fc2']:>8.4f} {j_m_mr['global']:>8.4f}")

    # Save Jaccard results
    jac_path = f"{OUT_PREFIX}_jaccard.json"
    with open(jac_path, 'w') as f:
        json.dump({
            'fisher_vs_magnitude':    j_f_m,
            'fisher_vs_mag_rewind':   j_f_mr,
            'magnitude_vs_mag_rewind': j_m_mr,
            'final_accuracy': {'fisher': acc_fisher, 'magnitude': acc_mag, 'mag_rewind': acc_mag_rew},
            'final_sparsity':  {'fisher': sp_fisher,  'magnitude': sp_mag,  'mag_rewind': sp_mag_rew},
        }, f, indent=2)
    print(f"\nJaccard results saved: {jac_path}")

    # Save round logs
    for name, log in [('fisher', log_fisher), ('magnitude', log_mag), ('mag_rewind', log_mag_rew)]:
        logpath = f"{OUT_PREFIX}_{name}_log.csv"
        with open(logpath, 'w', newline='') as f:
            if log:
                w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
                w.writeheader()
                w.writerows(log)

    # ── Plots ──────────────────────────────────────────────────────────────────
    COLORS = {'fisher': '#2196F3', 'magnitude': '#F44336', 'mag_rewind': '#4CAF50'}
    LABELS = {'fisher': 'Fisher/OBD', 'magnitude': 'Magnitude', 'mag_rewind': 'Mag + Rewind'}

    # 1. Accuracy vs sparsity across rounds for all three
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')
    for name, log in [('fisher', log_fisher), ('magnitude', log_mag), ('mag_rewind', log_mag_rew)]:
        sp  = [r['sparsity'] for r in log]
        acc = [r['acc']      for r in log]
        axes[0].plot(sp, acc, 'o-', color=COLORS[name], label=LABELS[name], lw=2, ms=5)
    axes[0].set_xlabel('Sparsity')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Accuracy vs Sparsity (all three methods)')
    axes[0].legend()
    axes[0].set_facecolor('white')
    axes[0].grid(True, alpha=0.3)

    # 2. Jaccard heatmap (3×3 symmetric, per method pair)
    methods = ['fisher', 'magnitude', 'mag_rewind']
    jac_matrix_global = np.ones((3, 3))
    jac_matrix_fc1    = np.ones((3, 3))
    jac_matrix_fc2    = np.ones((3, 3))
    pairs = [(0, 1, j_f_m), (0, 2, j_f_mr), (1, 2, j_m_mr)]
    for i, j, jdict in pairs:
        jac_matrix_global[i, j] = jac_matrix_global[j, i] = jdict['global']
        jac_matrix_fc1[i, j]    = jac_matrix_fc1[j, i]    = jdict['fc1']
        jac_matrix_fc2[i, j]    = jac_matrix_fc2[j, i]    = jdict['fc2']

    im = axes[1].imshow(jac_matrix_global, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_xticks(range(3))
    axes[1].set_yticks(range(3))
    short_labels = ['Fisher', 'Magnitude', 'Mag+\nRewind']
    axes[1].set_xticklabels(short_labels)
    axes[1].set_yticklabels(short_labels)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f'{jac_matrix_global[i,j]:.3f}',
                         ha='center', va='center', fontsize=12,
                         color='white' if jac_matrix_global[i,j] > 0.6 else 'black')
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title('Global Mask Jaccard Similarity')
    axes[1].set_facecolor('white')

    fig.suptitle(f'LeNet-300-100 MNIST: Mask Comparison at ~99% Sparsity', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plot_path = f"{OUT_PREFIX}_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Summary plot saved: {plot_path}")

    # 3. Per-layer mask overlap bars
    fig2, ax = plt.subplots(figsize=(9, 5), facecolor='white')
    pair_labels = ['Fisher\nvs Mag', 'Fisher\nvs Mag+Rew', 'Mag\nvs Mag+Rew']
    x = np.arange(len(pair_labels))
    w = 0.25
    fc1_vals   = [j_f_m['fc1'],   j_f_mr['fc1'],   j_m_mr['fc1']]
    fc2_vals   = [j_f_m['fc2'],   j_f_mr['fc2'],   j_m_mr['fc2']]
    glob_vals  = [j_f_m['global'],j_f_mr['global'],j_m_mr['global']]
    ax.bar(x - w,   fc1_vals,  w, label='fc1 (784→300)', color='#42A5F5', alpha=0.9)
    ax.bar(x,       fc2_vals,  w, label='fc2 (300→100)', color='#EF5350', alpha=0.9)
    ax.bar(x + w,   glob_vals, w, label='Global',        color='#66BB6A', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels)
    ax.set_ylabel('Jaccard Similarity')
    ax.set_ylim(0, 1.05)
    ax.set_title('Per-Layer Mask Overlap Between Methods (~99% Sparsity)')
    ax.legend()
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.3, axis='y')
    for bars, vals in [(x - w, fc1_vals), (x, fc2_vals), (x + w, glob_vals)]:
        for xi, v in zip(bars, vals):
            ax.text(xi, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    layer_plot_path = f"{OUT_PREFIX}_layerwise.png"
    plt.savefig(layer_plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Layer plot saved: {layer_plot_path}")


if __name__ == '__main__':
    main()
