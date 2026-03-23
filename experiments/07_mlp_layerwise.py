"""
MLP Layerwise Experiment: Test phase transition per layer.

Architecture: 4 inputs -> 8 hidden (ReLU) -> 4 hidden (ReLU) -> 1 output
N_total = 4*8 + 8*4 + 4*1 = 68 weights (tractable for Glauber)
M=200 samples for training
"""
import numpy as np
import os

from pruning_core.energy_mlp import (
    mlp_forward, mlp_loss, mlp_total_energy,
    grad_mlp_loss_w, mlp_glauber_step, relu, tanh_act,
    mlp_sample
)
from pruning_core.optimizers import AdamOptimizer


def run_glauber_mlp(w_init_list, h_init_list, X, y, eta_list, rho_list, alpha, T=50, activation='relu', rng=None):
    """
    Glauber dynamics for multi-layer network.
    
    - For each layer l, sweep coordinates in random order
    - Flip h[l][j,k] if ΔE < 0 (same MAP rule as perceptron)
    - After each full layer sweep, re-optimize all w via Adam
    
    Args:
        w_init_list: initial weights per layer
        h_init_list: initial masks per layer
        X: inputs (M, N_in)
        y: targets (M,)
        eta_list: eta per layer
        rho_list: rho per layer
        alpha: double-well barrier
        T: max iterations
        activation: activation function
        rng: random number generator
    
    Returns:
        dict with w, h, losses, history
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Initialize
    w = [w.copy() for w in w_init_list]
    h = [h.copy() for h in h_init_list]
    
    # Initial Adam optimization
    w = adam_optimize_mlp(w, h, X, y, eta_list, K=100, lr=1e-2)
    
    losses = []
    history = {'w': [w_copy(w) for w in w], 'h': [h.copy() for h in h], 'flips': []}
    
    it = 0
    E_diff = 1.0
    
    while E_diff > 0.0001 and it < T:
        # Run one Glauber sweep
        w, h, flips = mlp_glauber_step(w, h, X, y, eta_list, alpha, rho_list, activation, rng)
        
        # Optimize w with Adam
        w = adam_optimize_mlp(w, h, X, y, eta_list, K=20, lr=1e-2)
        
        # Compute current energy
        E_current = mlp_total_energy(w, h, X, y, eta_list, alpha, rho_list, activation)
        losses.append(E_current)
        
        if it > 0:
            E_diff = losses[-2] - losses[-1]
        
        history['w'].append(w_copy(w))
        history['h'].append([h.copy() for h in h])
        history['flips'].append(flips)
        
        it += 1
    
    return {
        'w': w,
        'h': h,
        'losses': losses,
        'history': history,
        'iterations': it
    }


def adam_optimize_mlp(w_list, h_list, X, y, eta_list, K=50, lr=1e-2):
    """
    Adam optimization for MLP weights.
    
    Args:
        w_list: list of weight matrices
        h_list: list of mask matrices
        X: inputs
        y: targets
        eta_list: eta per layer
        K: number of Adam steps
        lr: learning rate
    
    Returns:
        optimized weight list
    """
    # Initialize Adam
    params = [(w.flatten(),) for w in w_list]
    ms = [np.zeros_like(p[0]) for p in params]
    vs = [np.zeros_like(p[0]) for p in params]
    
    best_w = [w.copy() for w in w_list]
    best_E = float('inf')
    
    for _ in range(K):
        grad_list = grad_mlp_loss_w(w_list, h_list, X, y)
        
        # Add regularization gradient
        for l in range(len(w_list)):
            grad_list[l] = grad_list[l] + eta_list[l] * w_list[l]
        
        # Adam update per layer
        for l in range(len(w_list)):
            w_flat = w_list[l].flatten()
            g = grad_list[l].flatten()
            
            ms[l] = 0.9 * ms[l] + 0.1 * g
            vs[l] = 0.99 * vs[l] + 0.01 * g ** 2
            
            # Bias correction
            m_hat = ms[l] / (1 - 0.9 ** (_ + 1))
            v_hat = vs[l] / (1 - 0.99 ** (_ + 1))
            
            # Parameter update
            w_flat = w_flat - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        
        w_list = [p.reshape(w_list[l].shape) for l, p in enumerate(w_flat)]
        
        # Track best
        E = mlp_total_energy(w_list, h_list, X, y, eta_list, 1.0, rho_list)
        if E < best_E:
            best_E = E
            best_w = [w.copy() for w in w_list]
    
    return [w.copy() for w in best_w]


def w_copy(w_list):
    """Deep copy weight list."""
    return [w.copy() for w in w_list]


def layer_wise_hamming(h_list, h0_list):
    """Compute Hamming distance per layer."""
    hamming_per_layer = []
    for h, h0 in zip(h_list, h0_list):
        hamming_per_layer.append(np.sum((h - h0) ** 2) / h.size)
    return hamming_per_layer


def sparsity_per_layer(h_list):
    """Sparsity (fraction of zeros) per layer."""
    return [np.mean(h == 0) for h in h_list]


def main():
    np.random.seed(42)
    
    # Architecture: 4->8->4->1
    layer_sizes = [4, 8, 4, 1]
    M = 200  # training samples
    
    # Generate data
    print(f"Generating MLP data: {layer_sizes}, M={M}...")
    X, y, w0_list, h0_list = mlp_sample(M, layer_sizes, p0=0.5, sigma=0.01, seed=9900)
    
    # Print true sparsity per layer
    print("True masks:")
    for l, h0 in enumerate(h0_list):
        print(f"  Layer l: sparsity={np.mean(h0 == 0):.2%}")
    print()
    
    os.makedirs("results", exist_ok=True)
    
    # Parameters
    alpha = 1.0
    eta_val = 0.0001
    eta_list = [eta_val] * (len(layer_sizes) - 1)  # same for all layers
    rho_val = 0.0005
    rho_list = [rho_val] * (len(layer_sizes) - 1)
    
    # Run grid
    results = []
    
    # Test several (eta, rho) combinations
    eta_set = [0.0001, 0.0005]
    rho_set = [0.0001, 0.0005, 0.001]
    
    print("Running layerwise Glauber experiments...")
    print()
    
    for eta in eta_set:
        for rho in rho_set:
            print(f"eta={eta}, rho={rho}")
            
            # Re-initialize
            w_list = [w.copy() for w in w0_list]
            h_list = [h.copy() for h in h0_list]
            w_list = [np.ones_like(h) for h in h0_list]  # full initial mask
            
            # Run Glauber
            result = run_glauber_mlp(
                w_list, h_list, X, y,
                [eta] * (len(layer_sizes) - 1),
                [rho] * (len(layer_sizes) - 1),
                alpha, T=50, activation='relu'
            )
            
            h_final = result['h']
            
            # Compute per-layer metrics
            hamming_per_layer = layer_wise_hamming(h_final, h0_list)
            sparsity_per_layer = [np.mean(h == 0) for h in h_final]
            
            print(f"  Final: {sparsity_per_layer}")
            print(f"  Hamming per layer: {hamming_per_layer}")
            
            results.append({
                'eta': eta,
                'rho': rho,
                'layer_0_Hamming': hamming_per_layer[0],
                'layer_1_Hamming': hamming_per_layer[1],
                'layer_2_Hamming': hamming_per_layer[2],
            })
            
            print()
    
    # Write results
    csv_path = "results/mlp_layerwise.csv"
    with open(csv_path, 'w') as f:
        f.write("eta,rho,layer,Hamming,sparsity\n")
        for r in results:
            for l in range(3):
                Hamming_str = str(r[f'layer_{l}_Hamming'])
                # Get sparsity for this layer (need to recompute from result)
                f.write(f"{r['eta']},{r['rho']},{l},{Hamming_str},N/A\n")
    
    print(f"\nResults saved to {csv_path}")
    return results


if __name__ == "__main__":
    main()
