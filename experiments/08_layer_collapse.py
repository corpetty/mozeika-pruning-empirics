"""
Layer Collapse Experiment: Per-layer Hamming and active neuron fraction.

Architecture: 4->8->4->1 MLP, treated as stacked independent layers for Glauber.
Sweep rho to observe how layer-wise activity collapses.

Output: results/layer_collapse.csv
Columns: rho, layer, active_fraction, Hamming
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pruning_core.optimizers import AdamOptimizer
from pruning_core.metrics import hamming_distance


def relu(x):
    return np.maximum(0, x)


def generate_mlp_data(M, layer_sizes, p0=0.5, seed=42):
    """Generate MLP data with sparse true weights."""
    rng = np.random.default_rng(seed)
    w0_list = []
    h0_list = []
    
    dims = [(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
    
    for (n_in, n_out) in dims:
        w0 = rng.standard_normal((n_in, n_out)) * 0.5
        h0 = (rng.random((n_in, n_out)) < p0).astype(float)
        w0_list.append(w0)
        h0_list.append(h0)
    
    # Forward pass with true weights
    X = rng.standard_normal((M, layer_sizes[0]))
    a = X
    for w, h in zip(w0_list, h0_list):
        a = relu(a @ (w * h))
    y = a  # final layer output
    
    return X, y, w0_list, h0_list


def layer_energy(w, h, X_in, y_out, eta, rho, alpha=1.0):
    """Energy for a single layer: loss + L2 + double-well."""
    pred = relu(X_in @ (w * h))
    M = X_in.shape[0]
    loss = 0.5 * np.mean((pred - y_out) ** 2)
    l2 = 0.5 * eta * np.sum(w ** 2)
    dw = alpha * h**2 * (h - 1)**2 + 0.5 * rho * h**2
    return loss + l2 + np.sum(dw)


def optimize_layer_w(w, h, X_in, y_out, eta, K=20):
    """Adam steps on w with h fixed for one layer."""
    shape = w.shape
    opt = AdamOptimizer(N=w.size, lr=0.01)
    M = X_in.shape[0]
    w = w.copy()
    for _ in range(K):
        z = X_in @ (w * h)
        pred = relu(z)
        err = (pred - y_out) / M
        relu_mask = (z > 0).astype(float)
        err_pre = err * relu_mask
        grad = X_in.T @ err_pre  # (n_in, n_out)
        grad = grad * h + eta * w
        # AdamOptimizer expects flat arrays
        w_flat = opt.step(w.flatten(), grad.flatten())
        w = w_flat.reshape(shape)
    return w


def glauber_layer(w, h, X_in, y_out, eta, rho, alpha=1.0, T=30, rng=None):
    """Glauber dynamics on one layer."""
    if rng is None:
        rng = np.random.default_rng(0)
    w = w.copy()
    h = h.copy()
    n_in, n_out = w.shape
    
    for _ in range(T):
        # Sweep all weight positions
        indices = rng.permutation(n_in * n_out)
        for idx in indices:
            i, j = divmod(idx, n_out)
            old_h = h[i, j]
            new_h = 1.0 - old_h
            h[i, j] = new_h
            w_new = optimize_layer_w(w, h, X_in, y_out, eta, K=10)
            E_new = layer_energy(w_new, h, X_in, y_out, eta, rho, alpha)
            h[i, j] = old_h
            E_old = layer_energy(w, h, X_in, y_out, eta, rho, alpha)
            if E_new < E_old:
                h[i, j] = new_h
                w = w_new
        
        # Full re-optimize after sweep
        w = optimize_layer_w(w, h, X_in, y_out, eta, K=20)
    
    return w, h


def active_fraction(h):
    """Fraction of output neurons with at least one active incoming weight."""
    n_in, n_out = h.shape
    active = np.any(h > 0.5, axis=0).sum()
    return active / n_out


def main():
    np.random.seed(42)
    layer_sizes = [4, 8, 4, 1]
    M = 200
    p0 = 0.5
    alpha = 1.0
    eta = 0.0001
    
    rho_values = np.linspace(0, 0.005, 10)
    
    X, y_final, w0_list, h0_list = generate_mlp_data(M, layer_sizes, p0=p0, seed=42)
    
    os.makedirs("results", exist_ok=True)
    rows = []
    
    # For each layer, compute intermediate activations from true network
    # then run layerwise Glauber treating each layer independently
    for ri, rho in enumerate(rho_values):
        print(f"\nrho={rho:.5f}")
        rng = np.random.default_rng(ri * 1000)
        
        # Compute true intermediate outputs
        a = X
        true_intermediates = [X]
        for w0, h0 in zip(w0_list, h0_list):
            a = relu(a @ (w0 * h0))
            true_intermediates.append(a)
        
        for li in range(len(w0_list)):
            n_in, n_out = w0_list[li].shape
            w_init = np.random.randn(n_in, n_out) * 0.5
            h_init = np.ones((n_in, n_out))
            
            X_in = true_intermediates[li]
            y_out = true_intermediates[li + 1]
            
            w_final, h_final = glauber_layer(
                w_init, h_init, X_in, y_out,
                eta=eta, rho=rho, alpha=alpha, T=20, rng=rng
            )
            
            h0 = h0_list[li]
            N_l = n_in * n_out
            hd = hamming_distance(h_final.flatten().astype(int), h0.flatten().astype(int))
            af = active_fraction(h_final)
            
            print(f"  Layer {li}: Hamming={hd:.3f} active_frac={af:.3f}")
            rows.append((rho, li, af, hd))
    
    # Write CSV
    with open("results/layer_collapse.csv", "w") as f:
        f.write("rho,layer,active_fraction,Hamming\n")
        for rho, layer, af, hd in rows:
            f.write(f"{rho},{layer},{af:.4f},{hd:.4f}\n")
    
    print("\nSaved results/layer_collapse.csv")


if __name__ == "__main__":
    main()
