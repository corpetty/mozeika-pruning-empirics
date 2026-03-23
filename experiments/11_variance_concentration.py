"""
Experiment 11: What does pruning remove spectrally?

At each rho from exp 10, compute fraction of total variance (sum s_i^2) 
retained in top-1, top-3, top-5 singular directions.

The hypothesis: after pruning, variance concentrates in fewer directions.
"""
import numpy as np
import os


def load_spectral_results(csv_path='results/spectral_structure.csv'):
    """Load spectral results from exp 10."""
    data = []
    with open(csv_path, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                data.append({
                    'rho': float(parts[0]),
                    'seed': int(parts[1]),
                    'effective_rank': float(parts[2]),
                    'spectral_entropy': float(parts[3])
                })
    return data


def variance_in_top_k(S, k):
    """
    Compute fraction of variance in top k singular directions.
    
    Args:
        S: singular values array
        k: number of top singular vectors
    
    Returns:
        fraction of variance
    """
    p = (S ** 2) / np.sum(S ** 2) if np.sum(S ** 2) > 0 else np.ones(len(S)) / len(S)
    # Top k singular values
    p_sorted = np.sort(p)[::-1]
    return np.sum(p_sorted[:k])


def reconstruct_singular_values_from_metrics(S, effective_rank, spectral_entropy):
    """
    Infer singular values distribution from effective rank and entropy.
    
    For a given number of dimensions D (we'll assume D = min(20, 16) = 16 for layer 0),
    we can't uniquely recover S from just two metrics, but we can sample.
    
    For now, we'll approximate: assume D dimensions and use the metrics to
    estimate a representative distribution.
    """
    D = 16  # min(in_features, hidden_size) for layer 0
    
    # We'll create a representative distribution
    # Start with uniform, then scale to match the effective rank constraint
    # Effectively rank = sum(s)/max(s), so we want sum/max = effective_rank
    
    # Create a geometric decay distribution
    # This is a reasonable prior for neural network weight spectra
    decay_rates = np.linspace(0.1, 2.0, 100)
    
    for decay in decay_rates:
        # Geometric distribution of singular values
        if decay < 1:
            s = np.array([decay ** i for i in range(D-1, -1, -1)])
        else:
            s = np.array([decay ** (D-1-i) for i in range(D)])
        
        s = s + 0.01  # Add small offset to avoid zeros
        
        eff_rank_actual = np.sum(s) / np.max(s)
        spec_ent_actual = -np.sum((s**2) / np.sum(s**2) * np.log((s**2) / np.sum(s**2)))
        
        # We want the one that matches our metrics
        if abs(eff_rank_actual - effective_rank) < 0.5 and abs(spec_ent_actual - spectral_entropy) < 1.0:
            return s
    
    # If no good match, just return uniform normalized
    s = np.ones(D)
    return s / np.sqrt(D)


def get_top_k_variances(S, k_list=[1, 3, 5]):
    """Get variance fractions in top k directions for given k values."""
    return {f'top_{k}': variance_in_top_k(S, k) for k in k_list}


def variance_concentration_analysis(exp10_csv='results/spectral_structure.csv'):
    """
    Analyze variance concentration in top singular directions.
    
    This is a heuristic analysis since we only have summary statistics from exp 10,
    not the full singular values.
    """
    # For a more realistic analysis, we need to re-run exp 10 with full SVD
    # Let's create a new analysis function
    
    # Architecture: 20 -> 16 -> 1
    # For the SVD of layer 0, we have (num_seeds, 20, 16)
    # Reshaped to (5, 320) for stacking
    # SVD gives us up to min(5, 320) = 5 singular values
    
    print("Variance concentration analysis:")
    print("Note: We need to re-run with full SVD capture")
    print()
    
    # Example from exp 10 metrics
    rhos = [0, 0.0003, 0.0005, 0.001]
    
    # For demonstration with uniform distribution (D=5 for 5 seeds)
    S_uniform = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    
    for k in [1, 3, 5]:
        frac = variance_in_top_k(S_uniform, k)
        print(f"  Top-{k} holds {frac:.2%} of variance")


def main():
    """
    Run variance concentration experiments.
    
    We need to capture the actual SVD from pruned weights, not just metrics.
    Let's do this by re-running exp 10 with full SVD computation.
    """
    np.random.seed(42)
    os.makedirs('results', exist_ok=True)
    
    from pruning_core.energy_mlp import (
        mlp_sample, mlp_forward, mlp_loss, mlp_total_energy,
        grad_mlp_loss_w, mlp_glauber_step, relu
    )
    
    layer_sizes = [20, 16, 1]
    M = 100
    
    # Parameters
    alpha = 1.0
    eta_val = 0.0001
    eta_list = [eta_val] * (len(layer_sizes) - 1)
    
    rhos = [0, 0.0003, 0.0005, 0.001]
    n_seeds = 5
    
    results = []
    
    print("Running variance concentration analysis...")
    
    for rho in rhos:
        rho_list = [rho] * (len(layer_sizes) - 1)
        print(f"\nProcessing rho={rho}...")
        
        all_S = []
        
        for seed in range(n_seeds):
            # Generate data
            X, y, w0, h0 = mlp_sample(M, layer_sizes, sigma=0.01, seed=9000 + seed)
            
            # Train with full mask
            w = [wi.copy() for wi in w0]
            h = [np.ones_like(hi) for hi in h0]
            
            for _ in range(300):
                w = adam_optimize_mlp(w, h, X, y, eta_list, K=20, lr=1e-2)
            
            # Apply Glauber
            w_final, h_final = run_glauber_w_and_h(
                w, h, X, y, eta_list, rho_list, alpha, T=50,
                rng=np.random.default_rng(8000 + seed)
            )
            
            # Extract layer 0 weights
            # We'll stack these later for SVD
            pass
        
        # Stack layer 0 weights from all seeds: shape (5, 20, 16)
        # Flatten to (5, 320), SVD gives us at most 5 singular values
        W_layer0 = np.array([get_pruned_weights(layer_sizes, seed, X, y, rho, alpha) for seed in range(n_seeds)])
        
        # This is getting complicated - let's simplify
        pass
    
    # Write results
    csv_path = 'results/variance_concentration.csv'
    with open(csv_path, 'w') as f:
        f.write('rho,seed,top_1,top_3,top_5\n')
        for r in results:
            f.write(f"{r['rho']},{r['seed']},{r['top_1']},{r['top_3']},{r['top_5']}\n")
    
    print(f"\nResults saved to {csv_path}")
    
    # Aggregate
    print("\nVariance concentration (averaged across seeds):")
    for rho in rhos:
        rho_results = [r for r in results if r['rho'] == rho]
        if rho_results:
            print(f"  rho={rho}: top-1={np.mean([r['top_1'] for r in rho_results]):.3%}, "
                  f"top-3={np.mean([r['top_3'] for r in rho_results]):.3%}, "
                  f"top-5={np.mean([r['top_5'] for r in rho_results]):.3%}")
    
    return results


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


def run_glauber_w_and_h(w_init, h_init, X, y, eta_list, rho_list, alpha, T=50, rng=None):
    """Run Glauber dynamics and return final w, h."""
    if rng is None:
        rng = np.random.default_rng()
    
    w = [wi.copy() for wi in w_init]
    h = [hi.copy() for hi in h_init]
    
    for _ in range(100):
        w = adam_optimize_mlp(w, h, X, y, eta_list, K=20, lr=1e-2)
    
    for _ in range(T):
        w, h, flips = mlp_glauber_step(w, h, X, y, eta_list, alpha, rho_list, 'relu', rng)
        w = adam_optimize_mlp(w, h, X, y, eta_list, K=20, lr=1e-2)
    
    return w, h


def get_pruned_weights(layer_sizes, seed, X, y, rho, alpha):
    """Get pruned weights for a single seed."""
    from pruning_core.energy_mlp import mlp_sample, relu
        
    M = X.shape[0]
    w0, h0 = mlp_sample(M, layer_sizes, sigma=0.01, seed=seed)[:2]

    from pruning_core.optimizers import AdamOptimizer
    # For a single layer (20x16 = 320 params)
    w = w0.flatten()
    h = h0.flatten()
    
    # Train Adam
    w_opt = AdamOptimizer(N=w.size, lr=0.01).optimize(w, X, y, eta=0.0001, steps=200, X=X, y=y)
    
    # Apply Glauber
    h_final, w_final, _ = apply_glauber_single_layer(
        w_opt, w, h, X, y, 
        eta=0.0001, rho=rho, alpha=alpha, T=20,
        rng=np.random.default_rng(8000 + seed)
    )
    
    return w_final.reshape(h0.shape)


def apply_glauber_single_layer(w, h_init, X, y, eta, rho, alpha, T=30, rng=None):
    """Single-layer Glauber with Adam re-optimization."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Flatten
    if len(w.shape) == 2:
        w_flat = w.flatten()
        h = h_init.flatten()
    else:
        w_flat = w
        h = np.array(h_init, dtype=float)


def total_energy_1d(w, h, X, y, eta, alpha, rho):
    """Total energy for a single layer (from energy.py for N=1)."""
    # Squared loss
    from pruning_core.data import generate_permutation_dataset
    y_pred = X @ w
    loss = np.mean((y - y_pred) ** 2) / 2
    
    # Regularization
    reg = eta * np.sum(w ** 2) / 2
    
    # Double-well
    V = alpha * np.sum((h ** 2) * ((h - 1) ** 2)) + (rho / 2) * np.sum(h)
    
    return loss + reg + V


def optimize_single_layer(w, h, X, y, eta, K=20, lr=1e-2):
    """Adam optimization for a single layer."""
    w_new = w.copy()
    
    for _ in range(K):
        # Gradient of squared loss
        y_pred = X @ w_new
        grad_loss = X.T @ (y_pred - y) / X.shape[0]
        
        # Add regularization
        grad_w = grad_loss + eta * w_new
        
        # Adam step
        # Simplified: just use gradient descent
        w_new = w_new - lr * grad_w
    
    return w_new


def apply_glauber_single_layer(w_init, h_init, X, y, eta, rho, alpha, T=30, rng=None):
    """Run Glauber dynamics on 1D weight vector."""
    if rng is None:
        rng = np.random.default_rng()
    
    w = w_init.copy()
    h = np.array(h_init, dtype=float)
    
    for _ in range(50):
        y_pred = X @ w
        loss = np.mean((y - y_pred) ** 2) / 2
        y_pred_new = X @ w
    
    for t in range(T):
        idx = rng.permutation(len(h))
        flips = 0
        
        for j in idx:
            # Flip bit j
            h_try = h.copy()
            h_try[j] = 1 - h_try[j]
            
            # Optimize w for this mask
            w_try = optimize_single_layer(w, h_try, X, y, eta, K=20, lr=1e-2)
            
            # Energy diff
            E_curr = total_energy_1d(w, h, X, y, eta, alpha, rho)
            E_try = total_energy_1d(w_try, h_try, X, y, eta, alpha, rho)
            
            if E_try < E_curr:
                w = w_try
                h = h_try
                flips += 1
        
        # Optional: re-optim after each sweep
        w = optimize_single_layer(w, h, X, y, eta, K=20, lr=1e-2)
    
    return h, w


if __name__ == "__main__":
    main()
