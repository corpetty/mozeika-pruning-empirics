#!/usr/bin/env python3
"""
Experiment 28: Synthetic Linear Teacher — Controlled Mask Recovery Test

Goal: Validate that OBD saliency correctly recovers a sparse teacher mask
in the perceptron-equivalent setting where:
- Data is exactly Gaussian
- Network has LINEAR activations (no ReLU — eliminates permutation symmetry)
- W2 is frozen random (fixed), only W1 is masked
- Teacher has known sparse W1_true with p0=0.5 sparsity

Architecture (per spec):
- N_in = 100, N_h = 50, N_out = 1
- W2_true: random fixed (N_out x N_h = 1 x 50)
- W1_true: sparse (N_h x N_in = 50 x 100) with p0=0.5 sparsity
- Data: y = W2_true @ (W1_true * H1_true) @ X.T + noise (sigma=0.01)
- M_train = 3000 samples

Algorithm (THE CORRECT CYCLE — per spec):
For each rho in np.logspace(-6, 0, 25):
  For each seed in range(5):
    Initialize W1 randomly, H1 = ones (all active)
    For iteration in range(50):
      STEP 1 — Train W1 to convergence (H1 fixed)
      STEP 2 — Estimate Fisher (diagonal) using 50 batches
      STEP 3 — Compute OBD saliency: S_i = 0.5 * F_ii * w_i^2
      STEP 4 — Prune H1: set to 0 where S_i < rho/2
      STEP 5 — Fine-tune W1 for 100 steps
      Check: if no weights pruned → stop

Key insight: With LINEAR activations and fixed W2, the problem is
identifiable up to scaling. We measure Hamming distance on H1 only.

Author: Based on Mozeika's framework and ChatGPT reasoning session (2026-03-25)
"""

import numpy as np
import os


def create_teacher(N_in, N_h, N_out, p0=0.5, seed=42):
    """
    Create teacher with:
    - W2_true: random fixed (N_out x N_h)
    - W1_true: sparse (N_h x N_in) with p0 sparsity
    - H1_true: mask for W1 (1 if active, 0 if pruned)
    
    Returns:
        W1_true, W2_true, H1_true
    """
    np.random.seed(seed)
    
    # W2_true: random fixed (no sparsity, no mask)
    W2_true = np.random.randn(N_out, N_h)
    
    # W1_true: sparse with p0 sparsity
    H1_true = (np.random.rand(N_h, N_in) > p0).astype(float)
    W1_true = np.random.randn(N_h, N_in) * H1_true
    
    return W1_true, W2_true, H1_true


def generate_data(N_in, W1_true, W2_true, M, sigma_noise=0.01):
    """
    Generate synthetic data from teacher network.
    
    X: (M, N_in) Gaussian inputs
    y: (M, N_out) outputs with small noise
    y = W2_true @ (W1_true * H1_true) @ X.T + noise
    """
    np.random.seed(42)
    X = np.random.randn(M, N_in)
    z1 = X @ W1_true.T  # (M, N_h)
    y = z1 @ W2_true.T  # (M, N_out)
    y += np.random.randn(M, 1) * sigma_noise
    
    return X, y


def train_to_convergence(X, y, W1, H1, W2_true, lr=0.01, n_iter=500, eta=0.001, tol=1e-5):
    """
    Train W1 to local minimum with H1 fixed.
    
    Loss: L = ||y - W2_true @ (W1 * H1) @ X.T||^2 / (2M) + eta * ||W1 * H1||^2
    
    Returns:
        W1_trained, n_iterations
    """
    M = X.shape[0]
    
    def compute_loss(W1, H1, X, y, eta):
        """Compute loss with L2 regularization."""
        W1_eff = W1 * H1
        y_pred = X @ W1_eff.T @ W2_true.T  # (M, 1)
        residual = y_pred - y
        loss = np.mean(residual ** 2) + eta * np.sum(W1_eff ** 2)
        return loss
    
    def compute_grad(W1, H1, X, y, eta):
        """Compute gradient w.r.t. W1."""
        M = X.shape[0]
        N_h = H1.shape[0]
        
        W1_eff = W1 * H1
        
        # Forward
        z1 = X @ W1_eff.T  # (M, N_h)
        y_pred = z1 @ W2_true.T  # (M, 1)
        
        # Backward
        dy = 2 * (y_pred - y) / M  # (M, 1)
        dz1 = dy @ W2_true  # (M, N_h)
        dW1 = dz1.T @ X  # (N_h, N_in)
        
        # Add L2 gradient (only on active weights)
        dW1 *= H1
        dW1 += eta * W1_eff
        
        return dW1
    
    # Training loop
    for iteration in range(n_iter):
        dW1 = compute_grad(W1, H1, X, y, eta)
        W1 -= lr * dW1
        
        # Check convergence (gradient norm)
        grad_norm = np.sqrt(np.sum(dW1 ** 2))
        if grad_norm < tol:
            return W1, iteration + 1
    
    return W1, n_iter


def estimate_fisher(W1, H1, X, y, W2_true, n_batches=50, batch_size=32):
    """
    Estimate diagonal Fisher using 50 batches.
    
    F_ii = E[g_i^2] where g_i is gradient w.r.t. W1[i, :]
    """
    M = X.shape[0]
    n_batches = min(n_batches, M // batch_size)
    
    F = np.zeros_like(W1)
    
    for _ in range(n_batches):
        # Sample batch
        idx = np.random.choice(M, batch_size, replace=False)
        X_batch = X[idx]
        y_batch = y[idx]
        
        # Forward with current H1
        W1_eff = W1 * H1
        z1 = X_batch @ W1_eff.T
        y_pred = z1 @ W2_true.T
        
        # Backward (gradients)
        dy = (y_pred - y_batch).flatten() / batch_size
        dz1 = dy[:, np.newaxis] * W2_true  # (batch_size, N_h)
        dW1_batch = dz1.T @ X_batch  # (N_h, N_in)
        
        # Accumulate squared gradients
        F += dW1_batch ** 2
    
    F /= n_batches
    return F


def compute_obd_saliency(W1, H1, F):
    """
    Compute OBD saliency: S_i = 0.5 * F_ii * w_i^2
    """
    W1_eff = W1 * H1
    S = 0.5 * F * (W1_eff ** 2)
    return S


def prune_mask(H1, S, rho):
    """
    Prune mask: set to 0 where S_i < rho/2
    """
    H1_new = H1.copy()
    mask = (S < rho / 2.0)
    H1_new[mask] = 0
    return H1_new


def hamming_distance(mask1, mask2):
    """Compute Hamming distance between two masks."""
    return np.sum(mask1 != mask2) / mask1.size


def run_experiment(rho, seed, max_iter=50, n_train=3000, N_in=100, N_h=50, N_out=1):
    """
    Run one experiment with given rho and seed.
    
    Returns:
        results: dict with rho, seed, hamming_distances, active_fracs, n_iterations
    """
    np.random.seed(seed)
    
    # Create teacher
    W1_teacher, W2_true, H1_teacher = create_teacher(
        N_in, N_h, N_out, p0=0.5, seed=seed
    )
    
    # Generate data
    X, y = generate_data(N_in, W1_teacher, W2_true, M=n_train, sigma_noise=0.01)
    
    # Initialize student (all active)
    W1 = np.random.randn(N_h, N_in) * 0.1
    H1 = np.ones((N_h, N_in))
    
    # Training/pruning loop
    history = []
    for iteration in range(max_iter):
        # STEP 1 — Train W1 to convergence (H1 fixed)
        W1, n_iter_train = train_to_convergence(X, y, W1, H1, W2_true, lr=0.01, n_iter=500, eta=0.001)
        
        # STEP 2 — Estimate Fisher (50 batches)
        F = estimate_fisher(W1, H1, X, y, W2_true, n_batches=50, batch_size=32)
        
        # STEP 3 — Compute OBD saliency
        S = compute_obd_saliency(W1, H1, F)
        
        # STEP 4 — Prune
        H1_new = prune_mask(H1, S, rho)
        
        # STEP 5 — Fine-tune for 100 steps
        W1, _ = train_to_convergence(X, y, W1, H1_new, W2_true, lr=0.01, n_iter=100, eta=0.001)
        
        # Check convergence (no weights changed)
        if np.array_equal(H1_new, H1):
            H1 = H1_new
            break
        
        H1 = H1_new
        
        # Record metrics
        ham = hamming_distance(H1, H1_teacher)
        active_frac = H1.mean()
        
        history.append({
            'iteration': iteration,
            'ham': ham,
            'active_frac': active_frac
        })
    
    # Final metrics
    ham_final = hamming_distance(H1, H1_teacher)
    active_frac_final = H1.mean()
    
    return {
        'rho': rho,
        'seed': seed,
        'ham_final': ham_final,
        'active_frac_final': active_frac_final,
        'n_iterations': len(history),
        'history': history
    }


def main():
    """Run all experiments and save results."""
    # Configuration (per spec)
    rho_grid = np.logspace(-6, 0, 25)  # 25 values from 1e-6 to 1e0
    seeds = range(5)  # 5 seeds
    N_in, N_h, N_out = 100, 50, 1
    n_train = 3000
    
    print(f"Running Experiment 28: Synthetic Linear Teacher")
    print(f"  rho_grid: {len(rho_grid)} values from {rho_grid[0]:.2e} to {rho_grid[-1]:.2e}")
    print(f"  seeds: {len(seeds)}")
    print(f"  architecture: {N_in} -> {N_h} -> {N_out}")
    print(f"  teacher sparsity (p0): 0.5")
    print(f"  training samples: {n_train}")
    print(f"  Fisher batches: 50")
    print()
    
    results = []
    
    for i_rho, rho in enumerate(rho_grid):
        for seed in seeds:
            print(f"  [{i_rho+1}/{len(rho_grid)*len(seeds)}] rho={rho:.2e}, seed={seed}...")
            result = run_experiment(
                rho=rho,
                seed=seed,
                max_iter=50,
                n_train=n_train,
                N_in=N_in,
                N_h=N_h,
                N_out=N_out
            )
            results.append(result)
            
            # Print progress
            print(f"    Final Hamming: {result['ham_final']:.4f}")
            print(f"    Active fraction: {result['active_frac_final']:.4f}")
            print(f"    Iterations: {result['n_iterations']}")
    
    # Save to CSV
    import csv
    os.makedirs('results', exist_ok=True)
    
    with open('results/synthetic_linear.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rho', 'seed', 'ham_final', 'active_frac_final', 'n_iter_converged'])
        for r in results:
            writer.writerow([
                r['rho'],
                r['seed'],
                r['ham_final'],
                r['active_frac_final'],
                r['n_iterations']
            ])
    
    print(f"\nResults saved to: results/synthetic_linear.csv")
    print(f"Shape: {len(results)} rows")
    
    # Summary statistics
    print("\nSummary by rho:")
    for rho in rho_grid:
        subset = [r for r in results if r['rho'] == rho]
        if len(subset) > 0:
            avg_ham = np.mean([r['ham_final'] for r in subset])
            avg_active = np.mean([r['active_frac_final'] for r in subset])
            print(f"  rho={rho:.2e}: Hamming={avg_ham:.4f}, Active={avg_active:.4f}")


if __name__ == "__main__":
    main()