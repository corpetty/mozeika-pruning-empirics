"""
Experiment 12: Subspace convergence across seeds.

Use the 5 trained+pruned MLPs from exp 10.
For each pair of seeds (10 pairs), compute principal angles between
column spaces of pruned W_layer0 using scipy.linalg.subspace_angles.

Hypothesis: principal angle decreases near rho_c.
"""
import numpy as np
import os
import pandas as pd


def subspace_angle(A, B):
    """
    Compute principal angle between two subspaces.
    
    A: matrix of basis vectors for subspace 1 (M, k1)
    B: matrix of basis vectors for subspace 2 (M, k2)
    
    Returns: angle in radians (smallest principal angle)
    """
    # Use SVD to find the principal angles
    # The cosine of the smallest principal angle is the largest singular value of U^T V
    U1, S1, V1 = np.linalg.svd(A, full_matrices=False)
    U2, S2, V2 = np.linalg.svd(B, full_matrices=False)
    
    # Cosine of principal angles
    cos_angles = np.abs(U1.T @ U2)
    singular_vals = np.linalg.svd(cos_angles, compute_uv=False)
    
    # Principal angles are arccos(singular values)
    angles = np.arccos(np.clip(singular_vals, 0, 1))
    
    # Return the smallest (closest) angle
    return angles[0]


def subspace_angles_all(A, B, subset=3):
    """
    Compute principal angles between two subspaces.
    
    If both subspaces have dimension > subset, use only the top 'subset' dimensions.
    
    Returns array of principal angles (smallest first)
    """
    # If the subspaces are larger than 'subset', extract top 'subset' dimensions
    M1, k1 = A.shape
    M2, k2 = B.shape
    
    min_dim = min(k1, k2, subset) if k1 > subset or k2 > subset else min(k1, k2)
    
    if A.shape[1] > min_dim:
        A = A[:, :min_dim]
    if B.shape[1] > min_dim:
        B = B[:, :min_dim]
    
    # SVD to get orthonormal bases
    U1, S1, _ = np.linalg.svd(A, full_matrices=False)
    U2, S2, _ = np.linalg.svd(B, full_matrices=False)
    
    # Principal angles via SVD of U1.T @ U2
    cosines = U1.T @ U2
    singular_values = np.linalg.svd(cosines, compute_uv=False)
    
    angles = np.arccos(np.clip(singular_values, 0, 1))
    return angles


def get_pruned_weights_per_seed(layer_sizes, seed, M=100, alpha=1.0, eta_val=0.0001, 
                                 rho=0.001, T=50, activation='relu', rng=None):
    """
    Train MLP from seed and apply Mozeika pruning at rho.
    Return pruned layer 0 weights.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    from pruning_core.energy_mlp import mlp_sample, mlp_forward, mlp_loss, mlp_total_energy, \
        grad_mlp_loss_w, mlp_glauber_step, relu
    
    # Generate data
    X, y, w0, h0 = mlp_sample(M, layer_sizes, sigma=0.01, seed=9000 + seed)
    
    # Train from scratch with full mask
    eta_list = [eta_val] * (len(layer_sizes) - 1)
    rho_list = [rho] * (len(layer_sizes) - 1)
    
    w = [wi.copy() for wi in w0]
    h = [hi.copy() for hi in h0]
    
    # Optimize with full mask first
    for _ in range(100):
        w = adam_optimize_mlp(w, h, X, y, eta_list, K=20)
    
    # Apply Glauber dynamics
    for _ in range(T):
        w, h, flips = mlp_glauber_step(w, h, X, y, eta_list, alpha, rho_list, activation, rng)
        w = adam_optimize_mlp(w, h, X, y, eta_list, K=20)
    
    return w[0].copy()  # Return layer 0 weights


def adam_optimize_mlp(w_list, h_list, X, y, eta_list, K=50, lr=1e-2):
    """Adam optimization for MLP weights."""
    ms = [np.zeros_like(w.flatten()) for w in w_list]
    vs = [np.zeros_like(w.flatten()) for w in w_list]
    
    for _ in range(K):
        grad_list = grad_mlp_loss_w(w_list, h_list, X, y)
        for l in range(len(w_list)):
            grad_list[l] = grad_list[l] + eta_list[l] * w_list[l]
        
        for l in range(len(w_list)):
            w_flat = w_list[l].flatten()
            g = grad_list[l].flatten()
            
            ms[l] = 0.9 * ms[l] + 0.1 * g
            vs[l] = 0.99 * vs[l] + 0.01 * g ** 2
            
            m_hat = ms[l] / (1 - 0.9 ** (_ + 1))
            v_hat = vs[l] / (1 - 0.99 ** (_ + 1))
            
            w_new = w_flat - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            w_list[l] = w_new.reshape(w_list[l].shape)
    
    return w_list


def main():
    """
    Run subspace angle analysis.
    
    We need to extract the weight matrices from exp 10, then for each pair of seeds,
    compute the principal angle between their column spaces.
    """
    np.random.seed(42)
    os.makedirs('results', exist_ok=True)
    
    layer_sizes = [20, 16, 1]
    M = 100
    alpha = 1.0
    eta_val = 0.0001
    n_seeds = 5
    
    # Sparsity levels
    rhos = [0, 0.0003, 0.0005, 0.001]
    
    # First, collect pruned weights for each (seed, rho) pair
    print("Training and pruning MLPs...")
    weights_per_seed_rho = {}
    
    rng_main = np.random.default_rng(0)
    
    for rho in rhos:
        weights_per_seed_rho[rho] = []
        print(f"\n\rho={rho}:")
        
        for seed in range(n_seeds):
            w_layer0 = get_pruned_weights_per_seed(
                layer_sizes, seed, M=M, alpha=alpha, eta_val=eta_val,
                rho=rho, T=50, activation='relu', rng=rng_main.spawn(seed)
            )
            weights_per_seed_rho[rho].append(w_layer0)
            print(f"  Seed {seed}: weights shape={w_layer0.shape}")
    
    # Now for each rho, compute pairwise angles between subspaces
    print("\n\nComputing principal angles...")
    
    results = []
    
    for rho in rhos:
        W_all = np.array(weights_per_seed_rho[rho])  # (5, 20, 16)
        
        print(f"\nrho={rho}:")
        
        for seed1 in range(n_seeds):
            for seed2 in range(seed1 + 1, n_seeds):
                # Get the two weight matrices
                W1 = W_all[seed1]  # (20, 16)
                W2 = W_all[seed2]  # (20, 16)
                
                # Compute SVD
                U1, S1, V1 = np.linalg.svd(W1, full_matrices=False)
                U2, S2, V2 = np.linalg.svd(W2, full_matrices=False)
                
                # The column space of W is span(U) in R^20
                # Compute principal angles between the two subspaces
                n_common = min(len(S1), len(S2))
                
                # Cosine of principal angles
                cos_angles = np.abs(U1[:, :n_common].T @ U2[:, :n_common])
                singular_values = np.linalg.svd(cos_angles, compute_uv=False)
                angles_rad = np.arccos(np.clip(singular_values[:n_common], 0, 1))
                
                # Convert to degrees
                angles_deg = np.degrees(angles_rad)
                
                results.append({
                    'rho': rho,
                    'seed1': seed1,
                    'seed2': seed2,
                    'smallest_angle_deg': angles_deg[0],
                    'mean_angle_deg': np.mean(angles_deg),
                    'max_angle_deg': np.max(angles_deg)
                })
                
                if n_common <= 2:
                    print(f"  ({seed1},{seed2}): angle={angles_deg[0]:.2f}°")
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = 'results/subspace_angles.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Aggregate results
    print("\n=== Summary ===")
    for rho in rhos:
        rho_df = df[df['rho'] == rho]
        print(f"rho={rho}:")
        print(f"  Mean smallest angle: {rho_df['smallest_angle_deg'].mean():.2f}°")
        print(f"  Std smallest angle: {rho_df['smallest_angle_deg'].std():.2f}°")
    
    return results


if __name__ == "__main__":
    main()
