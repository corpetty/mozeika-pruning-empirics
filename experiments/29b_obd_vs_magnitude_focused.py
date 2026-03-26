#!/usr/bin/env python3
"""
Experiment 29b: OBD vs Magnitude on LeNet-300-100 + MNIST (focused run)

Single seed, single rho, multiple sparsity checkpoints.
Runtime target: ~20-30 min on single GPU.

Protocol (correct Mozeika cycle):
  1. Pretrain to convergence
  2. Repeat: estimate_diag_fisher → prune → fine-tune
  3. Record accuracy at sparsity checkpoints: 0%, 50%, 70%, 85%, 90%, 95%

Comparison: OBD saliency (S_i = 0.5 * F_ii * w_i^2) vs magnitude (|w_i|)
"""

import sys
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
PRETRAIN_EPOCHS = 5
FINETUNE_EPOCHS = 3
MAX_ROUNDS = 20
FISHER_BATCHES = 50
LR = 1e-3
SPARSITY_TARGETS = [0.0, 0.50, 0.70, 0.85, 0.90, 0.95]
SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Model ─────────────────────────────────────────────────────────────────────
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('weight_mask', torch.ones_like(self.weight))

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.weight_mask, self.bias)

    def apply_mask(self, mask):
        self.weight_mask.data.copy_(torch.from_numpy(mask).to(self.weight_mask.device))
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
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# ── Data ──────────────────────────────────────────────────────────────────────
def get_loaders():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2),
            DataLoader(test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2))


# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        for m in [model.fc1, model.fc2]:
            m.zero_masked_grads()
        optimizer.step()
        with torch.no_grad():
            for m in [model.fc1, model.fc2]:
                m.weight.data *= m.weight_mask


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def get_sparsity(model):
    total = active = 0
    for m in [model.fc1, model.fc2]:
        total  += m.weight_mask.numel()
        active += m.weight_mask.sum().item()
    return 1.0 - active / total


# ── Fisher diagonal ───────────────────────────────────────────────────────────
def estimate_fisher(model, loader, n_batches):
    """E[g^2] averaged over n_batches — diagonal Fisher proxy."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    F = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    batches = 0
    for x, y in loader:
        if batches >= n_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        model.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                F[n] += p.grad.data ** 2
        batches += 1
    for n in F:
        F[n] /= max(batches, 1)
    return F


# ── Pruning ───────────────────────────────────────────────────────────────────
def obd_prune(model, F, target_sparsity):
    """Prune to target_sparsity using OBD saliency S_i = 0.5 * F_ii * w_i^2."""
    layers = [('fc1', model.fc1), ('fc2', model.fc2)]
    all_sal, all_keys = [], []
    for name, layer in layers:
        w  = layer.weight.data.cpu().numpy().flatten()
        f  = F[f'{name}.weight'].cpu().numpy().flatten()
        mask = layer.weight_mask.data.cpu().numpy().flatten()
        sal  = 0.5 * f * w**2
        sal[mask == 0] = np.inf  # already pruned — don't touch
        all_sal.append(sal)
        all_keys.append((name, layer, sal.shape))

    combined = np.concatenate(all_sal)
    n_total  = sum(layer.weight_mask.numel() for _, layer in layers)
    n_prune  = max(0, int(target_sparsity * n_total) - int(get_sparsity(model) * n_total))
    if n_prune <= 0:
        return False

    threshold = np.partition(combined, n_prune)[n_prune]
    changed   = False
    for name, layer, _ in all_keys:
        w    = layer.weight.data.cpu().numpy()
        f    = F[f'{name}.weight'].cpu().numpy()
        sal  = 0.5 * f * w**2
        mask = layer.weight_mask.data.cpu().numpy()
        new_mask = np.where(sal < threshold, 0.0, mask)
        if not np.array_equal(new_mask, mask):
            changed = True
            layer.apply_mask(new_mask)
    return changed


def magnitude_prune(model, target_sparsity):
    """Prune to target_sparsity using |w| threshold."""
    layers = [('fc1', model.fc1), ('fc2', model.fc2)]
    all_abs = []
    for name, layer in layers:
        w    = layer.weight.data.cpu().numpy().flatten()
        mask = layer.weight_mask.data.cpu().numpy().flatten()
        abs_w = np.abs(w)
        abs_w[mask == 0] = np.inf
        all_abs.append(abs_w)

    combined = np.concatenate(all_abs)
    n_total  = sum(layer.weight_mask.numel() for _, layer in layers)
    n_prune  = max(0, int(target_sparsity * n_total) - int(get_sparsity(model) * n_total))
    if n_prune <= 0:
        return False

    threshold = np.partition(combined, n_prune)[n_prune]
    changed   = False
    for name, layer in layers:
        w    = layer.weight.data.cpu().numpy()
        mask = layer.weight_mask.data.cpu().numpy()
        new_mask = np.where(np.abs(w) < threshold, 0.0, mask)
        if not np.array_equal(new_mask, mask):
            changed = True
            layer.apply_mask(new_mask)
    return changed


# ── Main experiment ───────────────────────────────────────────────────────────
def run(method='obd'):
    print(f"\n{'='*60}")
    print(f"Method: {method.upper()}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}")

    train_loader, test_loader = get_loaders()
    model = LeNet300100().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Pretrain
    print(f"\nPretraining ({PRETRAIN_EPOCHS} epochs)...")
    for ep in range(PRETRAIN_EPOCHS):
        train_epoch(model, train_loader, optimizer, criterion)
        acc = evaluate(model, test_loader)
        print(f"  Epoch {ep+1}/{PRETRAIN_EPOCHS} — acc={acc:.4f}  sparsity={get_sparsity(model):.3f}")

    records = []
    sparsity = get_sparsity(model)
    acc = evaluate(model, test_loader)
    records.append({'method': method, 'sparsity_target': 0.0,
                    'achieved_sparsity': sparsity, 'test_acc': acc, 'round': 0})
    print(f"\nBaseline acc={acc:.4f}  sparsity={sparsity:.3f}")

    targets_remaining = [t for t in SPARSITY_TARGETS if t > 0]

    for rnd in range(1, MAX_ROUNDS + 1):
        if not targets_remaining:
            break
        target = targets_remaining[0]

        print(f"\nRound {rnd} — pruning to {target:.0%}...")

        if method == 'obd':
            F = estimate_fisher(model, train_loader, FISHER_BATCHES)
            changed = obd_prune(model, F, target)
        else:
            changed = magnitude_prune(model, target)

        if not changed:
            print(f"  No weights pruned at target {target:.0%}, skipping target")
            targets_remaining.pop(0)
            continue

        # Fine-tune
        optimizer = optim.Adam(model.parameters(), lr=LR)
        for ep in range(FINETUNE_EPOCHS):
            train_epoch(model, train_loader, optimizer, criterion)

        sparsity = get_sparsity(model)
        acc = evaluate(model, test_loader)
        print(f"  acc={acc:.4f}  achieved_sparsity={sparsity:.3f}  target={target:.0%}")

        records.append({'method': method, 'sparsity_target': target,
                        'achieved_sparsity': sparsity, 'test_acc': acc, 'round': rnd})

        if sparsity >= target - 0.02:
            targets_remaining.pop(0)

    return records


def main():
    os.makedirs('results', exist_ok=True)
    all_records = []
    all_records += run('obd')
    all_records += run('magnitude')

    outpath = 'results/obd_vs_magnitude_mnist.csv'
    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method','sparsity_target','achieved_sparsity','test_acc','round'])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n{'='*60}")
    print(f"Results saved to {outpath}")
    print(f"\n{'Method':<12} {'Target':>8} {'Achieved':>10} {'Acc':>8}")
    print('-'*42)
    for r in all_records:
        print(f"{r['method']:<12} {r['sparsity_target']:>8.0%} {r['achieved_sparsity']:>10.3f} {r['test_acc']:>8.4f}")


if __name__ == '__main__':
    main()
