#!/usr/bin/env python3
"""
Experiment 29: OBD vs Magnitude on LeNet-300-100 + MNIST

Goal: Compare OBD (Mozeika) vs Magnitude pruning on LeNet-300-100.

Architecture: LeNet-300-100 (2-layer MLP)
- Input: 784 (flattened MNIST)
- Hidden 1: 300 units with ReLU
- Hidden 2: 100 units with ReLU  
- Output: 10 (MNIST classes)

Methods:
1. OBD (Mozeika): Prune based on second-order Fisher approximation
2. Magnitude: Prune based on |w|

We run both methods with same sparsity levels and compare:
- Final accuracy
- Convergence speed
- Stability across seeds

Configuration:
- rho_grid: 25 values from 1e-6 to 1e0 (log space)
- seeds: 5 random seeds
- Training: SGD with momentum, learning rate decay
- Pruning: Iterative (50 iterations)
- Batch size: 128
- Fisher batches: 10 per iteration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import csv
import sys

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Configuration
N_SEEDS = 5
N_RHO = 25
MAX_ITER = 50
BATCH_SIZE = 128
FISHER_BATCHES = 10
LR_INIT = 0.01
MOMENTUM = 0.9
DECAY = 0.95

# Architecture
INPUT_DIM = 784
HIDDEN1_DIM = 300
HIDDEN2_DIM = 100
OUTPUT_DIM = 10

# Sparsity levels (rho values)
rho_grid = np.logspace(-6, 0, N_RHO)


class LeNet300100(nn.Module):
    """LeNet-300-100 architecture."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN1_DIM)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN1_DIM, HIDDEN2_DIM)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(HIDDEN2_DIM, OUTPUT_DIM)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def load_mnist(batch_size=BATCH_SIZE):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in loader:
        data = data.float().to(next(model.parameters()).device)
        target = target.long().to(next(model.parameters()).device)
        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data = data.float().to(next(model.parameters()).device)
            target = target.long().to(next(model.parameters()).device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    return accuracy


def compute_fisher(model, loader, n_batches=FISHER_BATCHES):
    """
    Compute diagonal Fisher approximation.
    
    F_ii = E[g_i^2] where g_i is gradient w.r.t. parameter i
    """
    model.eval()
    device = next(model.parameters()).device
    n_params = sum(p.numel() for p in model.parameters())
    F = np.zeros(n_params)
    
    # Get parameter indices
    param_indices = []
    param_shapes = []
    for name, p in model.named_parameters():
        param_indices.append((name, p))
        param_shapes.append(p.shape)
    
    print(f"    Computing Fisher for {n_params} params with {n_batches} batches...", flush=True)
    
    for batch_idx in range(n_batches):
        # Sample batch
        data, target = next(iter(loader))
        data = data.float().to(device)
        target = target.long().to(device)
        
        # Forward
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        
        # Backward
        model.zero_grad()
        loss.backward()
        
        # Accumulate squared gradients
        for idx, (name, p) in enumerate(param_indices):
            grad = p.grad.detach().cpu().numpy().flatten()
            F[idx * p.numel():(idx + 1) * p.numel()] += grad ** 2
        
        # Print progress
        if batch_idx % 2 == 0:
            print(f"    Fisher batch {batch_idx}/{n_batches}", flush=True)
    
    F /= n_batches
    print(f"    Fisher computed", flush=True)
    return F, param_indices, param_shapes


def compute_obd_saliency(model, F, param_indices, param_shapes):
    """
    Compute OBD saliency: S_i = 0.5 * F_ii * w_i^2
    """
    S = np.zeros_like(F)
    
    for idx, (name, p) in enumerate(param_indices):
        start = idx * p.numel()
        end = (idx + 1) * p.numel()
        w = p.detach().cpu().numpy().flatten()
        S[start:end] = 0.5 * F[start:end] * (w ** 2)
    
    return S


def prune_mask(model, S, param_indices, param_shapes, rho):
    """
    Prune mask: set to 0 where S_i < rho/2
    """
    mask = np.zeros_like(S)
    
    for idx, (name, p) in enumerate(param_indices):
        start = idx * p.numel()
        end = (idx + 1) * p.numel()
        mask[start:end] = (S[start:end] >= rho / 2.0).astype(float)
    
    return mask


def apply_mask(model, mask, param_indices, param_shapes):
    """Apply mask to model parameters."""
    device = next(model.parameters()).device
    for idx, (name, p) in enumerate(param_indices):
        start = idx * p.numel()
        end = (idx + 1) * p.numel()
        mask_tensor = torch.from_numpy(mask[start:end].reshape(p.shape)).to(device)
        p.data = p.data * mask_tensor


def train_to_convergence(model, loader, criterion, lr=0.01, momentum=0.9, n_epochs=100):
    """Train model to convergence."""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    for epoch in range(n_epochs):
        train_epoch(model, loader, optimizer, criterion)
        # Check convergence (loss change)
        if epoch % 10 == 0:
            avg_loss, _ = train_epoch(model, loader, optimizer, criterion)
            if avg_loss < 0.01:
                break
    
    return model, optimizer


def run_experiment(rho, seed, n_train=60000, n_val=10000):
    """
    Run one experiment with given rho and seed.
    
    Returns:
        results: dict with rho, seed, final_accuracy, active_params, n_iterations
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
    train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE)
    
    # Initialize model
    model = LeNet300100()
    device = next(model.parameters()).device
    
    # Initial training
    criterion = nn.CrossEntropyLoss()
    model, optimizer = train_to_convergence(
        model, train_loader, criterion, lr=LR_INIT, momentum=MOMENTUM, n_epochs=50
    )
    
    # Pruning loop
    history = []
    for iteration in range(MAX_ITER):
        # STEP 1 — Estimate Fisher
        F, param_indices, param_shapes = compute_fisher(model, train_loader, n_batches=FISHER_BATCHES)
        
        # STEP 2 — Compute OBD saliency
        S = compute_obd_saliency(model, F, param_indices, param_shapes)
        
        # STEP 3 — Prune
        mask = prune_mask(model, S, param_indices, param_shapes, rho)
        apply_mask(model, mask, param_indices, param_shapes)
        
        # STEP 4 — Fine-tune
        model, _ = train_to_convergence(
            model, train_loader, criterion, lr=LR_INIT * DECAY, momentum=MOMENTUM, n_epochs=20
        )
        
        # Evaluate
        acc = evaluate(model, test_loader)
        active = np.sum(mask) / mask.size
        
        history.append({
            'iteration': iteration,
            'accuracy': acc,
            'active_frac': active
        })
        
        # Check convergence
        if iteration > 0 and history[-1]['active_frac'] == history[-2]['active_frac']:
            break
    
    # Final evaluation
    final_acc = evaluate(model, test_loader)
    final_active = np.sum(mask) / mask.size
    
    return {
        'rho': rho,
        'seed': seed,
        'final_accuracy': final_acc,
        'active_params': final_active,
        'n_iterations': len(history),
        'history': history
    }


def run_magnitude_baseline(rho, seed):
    """
    Run magnitude pruning baseline.
    
    Same as OBD but uses |w| instead of OBD saliency.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
    train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE)
    
    # Initialize model
    model = LeNet300100()
    device = next(model.parameters()).device
    
    # Initial training
    criterion = nn.CrossEntropyLoss()
    model, optimizer = train_to_convergence(
        model, train_loader, criterion, lr=LR_INIT, momentum=MOMENTUM, n_epochs=50
    )
    
    # Pruning loop
    history = []
    for iteration in range(MAX_ITER):
        # STEP 1 — Compute magnitude saliency
        S = np.zeros(0)
        for p in model.parameters():
            S = np.concatenate([S, np.abs(p.detach().cpu().numpy().flatten())])
        
        # STEP 2 — Prune
        mask = (S >= rho / 2.0).astype(float)
        param_indices = [(name, p) for name, p in model.named_parameters()]
        param_shapes = [p.shape for p in model.parameters()]
        apply_mask(model, mask, param_indices, param_shapes)
        
        # STEP 3 — Fine-tune
        model, _ = train_to_convergence(
            model, train_loader, criterion, lr=LR_INIT * DECAY, momentum=MOMENTUM, n_epochs=20
        )
        
        # Evaluate
        acc = evaluate(model, test_loader)
        active = np.sum(mask) / mask.size
        
        history.append({
            'iteration': iteration,
            'accuracy': acc,
            'active_frac': active
        })
        
        # Check convergence
        if iteration > 0 and history[-1]['active_frac'] == history[-2]['active_frac']:
            break
    
    # Final evaluation
    final_acc = evaluate(model, test_loader)
    final_active = np.sum(mask) / mask.size
    
    return {
        'rho': rho,
        'seed': seed,
        'final_accuracy': final_acc,
        'active_params': final_active,
        'n_iterations': len(history),
        'history': history
    }


def main():
    """Run all experiments and save results."""
    print(f"Running Experiment 29: OBD vs Magnitude on LeNet-300-100 + MNIST", flush=True)
    print(f"  rho_grid: {N_RHO} values from {rho_grid[0]:.2e} to {rho_grid[-1]:.2e}", flush=True)
    print(f"  seeds: {N_SEEDS}", flush=True)
    print(f"  architecture: {INPUT_DIM} -> {HIDDEN1_DIM} -> {HIDDEN2_DIM} -> {OUTPUT_DIM}", flush=True)
    print(f"  training samples: {N_SEEDS * 60000}", flush=True)
    print(f"  Fisher batches: {FISHER_BATCHES}", flush=True)
    print()
    
    results_obd = []
    results_magnitude = []
    
    # Run OBD
    print("Running OBD...", flush=True)
    for i_rho, rho in enumerate(rho_grid):
        for seed in range(N_SEEDS):
            print(f"  [{i_rho+1}/{N_RHO*N_SEEDS}] rho={rho:.2e}, seed={seed}...", flush=True)
            result = run_experiment(rho=rho, seed=seed)
            results_obd.append(result)
            print(f"    Final Accuracy: {result['final_accuracy']:.4f}", flush=True)
    
    # Run Magnitude
    print("\nRunning Magnitude...", flush=True)
    for i_rho, rho in enumerate(rho_grid):
        for seed in range(N_SEEDS):
            print(f"  [{i_rho+1}/{N_RHO*N_SEEDS}] rho={rho:.2e}, seed={seed}...", flush=True)
            result = run_magnitude_baseline(rho=rho, seed=seed)
            results_magnitude.append(result)
            print(f"    Final Accuracy: {result['final_accuracy']:.4f}", flush=True)
    
    # Save to CSV
    os.makedirs('results', exist_ok=True)
    
    with open('results/obd_vs_magnitude_mnist.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'rho', 'seed', 'final_accuracy', 'active_params', 'n_iter_converged'])
        
        for r in results_obd:
            writer.writerow(['OBD', r['rho'], r['seed'], r['final_accuracy'], r['active_params'], r['n_iterations']])
        
        for r in results_magnitude:
            writer.writerow(['Magnitude', r['rho'], r['seed'], r['final_accuracy'], r['active_params'], r['n_iterations']])
    
    print(f"\nResults saved to: results/obd_vs_magnitude_mnist.csv", flush=True)
    print(f"Shape: {len(results_obd) + len(results_magnitude)} rows", flush=True)
    
    # Summary statistics
    print("\nSummary by rho:", flush=True)
    for rho in rho_grid:
        subset_obd = [r for r in results_obd if r['rho'] == rho]
        subset_mag = [r for r in results_magnitude if r['rho'] == rho]
        
        if len(subset_obd) > 0:
            avg_acc_obd = np.mean([r['final_accuracy'] for r in subset_obd])
            avg_active_obd = np.mean([r['active_params'] for r in subset_obd])
            print(f"  OBD rho={rho:.2e}: Acc={avg_acc_obd:.4f}, Active={avg_active_obd:.4f}", flush=True)
        
        if len(subset_mag) > 0:
            avg_acc_mag = np.mean([r['final_accuracy'] for r in subset_mag])
            avg_active_mag = np.mean([r['active_params'] for r in subset_mag])
            print(f"  Mag rho={rho:.2e}: Acc={avg_acc_mag:.4f}, Active={avg_active_mag:.4f}", flush=True)


if __name__ == "__main__":
    main()