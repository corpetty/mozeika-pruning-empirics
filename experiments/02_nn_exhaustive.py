"""
Exhaustive search over small network masks.

Architecture: 4 inputs → [4,4,2] hidden → 1 output
Total parameters N ≤ 20 for 2^N enumeration to be feasible.
"""
import numpy as np
from pruning_core import (
    squared_loss, total_energy, double_well,
    exhaustive_search, hamming_distance, mse_w
)


def sample_small_network_params(N, seed=None):
    """
    Sample true parameters for a small network.
    Architecture: 4 inputs → 2 hidden neurons → 1 output
    Total weights: (4*2 + 2*1) = 10 weights, 2 biases = 11 parameters
    Or 4→3→1: (4*3 + 3*1) = 15 weights, 4 biases = 19 parameters
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Architecture: 4 inputs → 3 hidden (tanh) → 1 output
    # Weights layer 1: 4x3 = 12
    # Bias layer 1: 3
    # Weights layer 2: 3x1 = 3
    # Bias layer 2: 1
    # Total: 19 parameters
    
    N_hidden1 = 3
    N_hidden2 = 1
    
    w1 = rng.standard_normal((4, N_hidden1))
    b1 = rng.standard_normal(N_hidden1)
    w2 = rng.standard_normal((N_hidden1, N_hidden2))
    b2 = rng.standard_normal(N_hidden2)
    
    # Flatten: [w1, b1, w2, b2]
    w0 = np.concatenate([w1.flatten(), b1, w2.flatten(), b2])
    
    # True mask: sparse with p0=0.5
    p0 = 0.5
    N1 = int(np.floor(len(w0) * p0))
    h0 = np.concatenate([np.ones(N1), np.zeros(len(w0) - N1)])
    rng.shuffle(h0)
    
    return w0, h0


def forward_net(x, w, h, N_w1=12, N_b1=3, N_w2=3, N_b2=1):
    """
    Forward pass for 4→3→1 network.
    
    Args:
        x: input (4,)
        w: flattened weights [w1, b1, w2, b2] (19,)
        h: mask (19,)
        N_w1, N_b1, N_w2, N_b2: layer dimensions
    """
    # Unflatten
    w1 = w[:N_w1].reshape(4, 3) * h[:N_w1].reshape(4, 3)
    b1 = w[N_w1:N_w1+N_b1] * h[N_w1:N_w1+N_b1]
    w2 = w[N_w1+N_b1:N_w1+N_b1+N_w2].reshape(3, 1) * h[N_w1+N_b1:N_w1+N_b1+N_w2].reshape(3, 1)
    b2 = w[N_w1+N_b1+N_w2:] * h[N_w1+N_b1+N_w2:]
    
    # Forward
    h1 = np.tanh(x @ w1 + b1)
    y = h1 @ w2 + b2
    
    return y


def forward_net_batch(X, w, h, N_w1=12, N_b1=3, N_w2=3, N_b2=1):
    """Forward pass for batch of inputs."""
    ys = np.array([forward_net(x, w, h, N_w1, N_b1, N_w2, N_b2) for x in X])
    return ys


def squared_loss_net(w, h, X, y, N_w1=12, N_b1=3, N_w2=3, N_b2=1):
    """Squared loss for network."""
    preds = forward_net_batch(X, w, h, N_w1, N_b1, N_w2, N_b2)
    return np.mean((y - preds) ** 2) / 2


def main():
    """Run exhaustive search on small network."""
    seed = 9999
    np.random.seed(seed)
    
    # Network architecture: 4→3→1
    N_w1, N_b1, N_w2, N_b2 = 12, 3, 3, 1
    N = N_w1 + N_b1 + N_w2 + N_b2  # 19
    
    print(f"Architecture: 4→3→1")
    print(f"Total parameters: N = {N}")
    print(f"Exhaustive search over 2^{N} = {2**N} masks")
    print()
    
    # Sample true parameters
    w0, h0 = sample_small_network_params(N, seed=seed)
    
    # Generate training data
    M = 100
    rng = np.random.default_rng(seed + 1)
    X = rng.standard_normal((M, 4))
    y = np.array([forward_net(x, w0, h0, N_w1, N_b1, N_w2, N_b2) for x in X])
    y += 0.01 * rng.standard_normal(M)  # small noise
    
    # Test data
    M_test = 1000
    X_test = rng.standard_normal((M_test, 4))
    y_test = np.array([forward_net(x, w0, h0, N_w1, N_b1, N_w2, N_b2) for x in X_test])
    
    print(f"Training data: M={M}, test data: M_test={M_test}")
    print()
    
    # Small grid for demonstration (full grid would be slow)
    eta_set = np.array([0.0, 0.0005, 0.001])
    rho_set = np.array([0.0, 0.0005, 0.001])
    
    alpha = 1.0
    
    # Results
    results = []
    
    for eli, eta in enumerate(eta_set):
        for rli, rho in enumerate(rho_set):
            print(f"eta={eta:.6f}, rho={rho:.6f}")
            
            # Exhaustive search
            result = exhaustive_search(
                X, y, eta, rho, alpha, N, 
                K_adam=20,  # Fewer steps for speed
                N_w1=N_w1, N_b1=N_b1, N_w2=N_w2, N_b2=N_b2
            )
            
            w_opt, h_opt, E_opt = result['w'], result['h'], result['E']
            
            # Metrics
            Hamming = np.sum((h_opt - h0) ** 2) / N
            MSE = np.sum((w_opt * h_opt - w0 * h0) ** 2) / N
            sparsity = 1 - np.mean(h_opt)
            
            # Train/test errors
            L_train = squared_loss_net(w_opt, h_opt, X, y, N_w1, N_b1, N_w2, N_b2)
            L_test = np.mean((y_test - forward_net_batch(X_test, w_opt, h_opt, N_w1, N_b1, N_w2, N_b2)) ** 2) / 2
            
            print(f"  E={E_opt:.4f}, Hamming={Hamming:.4f}, MSE={MSE:.6f}, "
                  f"sparsity={sparsity:.2%}, L_train={L_train:.4f}, L_test={L_test:.4f}")
            
            results.append({
                'eta': eta,
                'rho': rho,
                'E': E_opt,
                'Hamming': Hamming,
                'MSE': MSE,
                'sparsity': sparsity,
                'train_error': L_train,
                'test_error': L_test,
                'w': w_opt,
                'h': h_opt
            })
    
    # Save results
    import json
    
    results_dict = {
        'params': {
            'seed': seed,
            'N': N,
            'N_w1': N_w1,
            'N_b1': N_b1,
            'N_w2': N_w2,
            'N_b2': N_b2,
            'M': M,
            'M_test': M_test,
            'alpha': alpha,
            'eta_set': eta_set.tolist(),
            'rho_set': rho_set.tolist()
        },
        'results': [
            {k: v for k, v in r.items() if k != 'w' and k != 'h'}
            for r in results
        ]
    }
    
    with open('exhaustive_search_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("\nResults saved to exhaustive_search_results.json")
    return results


if __name__ == "__main__":
    # Override the exhaustive_search function to accept network-specific args
    import pruning_core.dynamics as dyn
    
    original_exhaustive = dyn.exhaustive_search
    
    def exhaustive_search_fixed(X, y, eta, rho, alpha, N, K_adam=50,
                                N_w1=12, N_b1=3, N_w2=3, N_b2=1):
        def loss_fn(w, h):
            return squared_loss_net(w, h, X, y, N_w1, N_b1, N_w2, N_b2)
        
        best_E = float('inf')
        best_w = None
        best_h = None
        
        for mask_idx in range(2 ** N):
            h = np.array([(mask_idx >> i) & 1 for i in range(N)], dtype=float)
            
            # Adam optimization
            w = np.random.randn(N)
            lr = 1e-2
            for k in range(K_adam):
                # Gradient for network
                w_h = w * h
                preds = forward_net_batch(X, w_h, h, N_w1, N_b1, N_w2, N_b2)
                residuals = preds - y
                # Simplified gradient (identity activation)
                grad_loss = (X.T @ residuals) / len(y) * h
                grad_reg = eta * w
                grad = grad_loss + grad_reg
                
                # Adam step
                # (simplified, no momentum for demo)
                w = w - lr * grad
            
            E = loss_fn(w, h) + (eta / 2) * np.sum(w ** 2) + double_well(h, alpha, rho)
            
            if E < best_E:
                best_E = E
                best_w = w.copy()
                best_h = h.copy()
        
        return {'w': best_w, 'h': best_h, 'E': best_E}
    
    dyn.exhaustive_search = exhaustive_search_fixed
    main()
    dynam.exhaustive_search = original_exhaustive  # Restore
