"""
Activation Comparison: identity vs tanh vs relu on perceptron Glauber.

N=100, M=200, sweep rho at fixed eta=0.0005.
Data always generated from identity network (linear).

Output: results/activation_comparison.csv
Columns: activation, rho, Hamming, iterations, energy
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pruning_core.optimizers import AdamOptimizer
from pruning_core.metrics import hamming_distance
from pruning_core.energy import total_energy, squared_loss


def identity(x): return x
def relu(x): return np.maximum(0, x)
def tanh_act(x): return np.tanh(x)

def identity_grad(x): return np.ones_like(x)
def relu_grad(x): return (x > 0).astype(float)
def tanh_grad(x): return 1 - np.tanh(x) ** 2


def optimize_w(w, h, X, y, eta, phi, phi_grad_fn, K=20):
    opt = AdamOptimizer(N=len(w), lr=0.01)
    M = X.shape[0]
    w = w.copy()
    for _ in range(K):
        z = X @ (w * h)
        pred = phi(z)
        err = (pred - y) / M
        grad_z = err * phi_grad_fn(z)
        grad_w = X.T @ grad_z
        grad_w = grad_w * h + eta * w
        w = opt.step(w, grad_w)
    return w


def activation_energy(w, h, X, y, eta, rho, phi, alpha=1.0):
    M = X.shape[0]
    pred = phi(X @ (w * h))
    loss = 0.5 * np.mean((pred - y) ** 2)
    l2 = 0.5 * eta * np.sum(w ** 2)
    dw = alpha * h**2 * (h - 1)**2 + 0.5 * rho * h**2
    return loss + l2 + np.sum(dw)


def glauber_activation(w0_init, h0_init, X, y, eta, rho, phi, phi_grad_fn, T=50, alpha=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    w = w0_init.copy()
    h = h0_init.copy()
    N = len(w)
    
    iters_to_converge = T
    for t in range(T):
        E_before = activation_energy(w, h, X, y, eta, rho, phi, alpha)
        changed = 0
        for j in rng.permutation(N):
            old_h = h[j]
            h[j] = 1.0 - old_h
            w_new = optimize_w(w, h, X, y, eta, phi, phi_grad_fn, K=10)
            E_new = activation_energy(w_new, h, X, y, eta, rho, phi, alpha)
            E_old = activation_energy(w, h, X, y, eta, rho, phi, alpha)
            if E_new < E_old:
                w = w_new
                changed += 1
            else:
                h[j] = old_h
        w = optimize_w(w, h, X, y, eta, phi, phi_grad_fn, K=20)
        if changed == 0:
            iters_to_converge = t + 1
            break
    
    E_final = activation_energy(w, h, X, y, eta, rho, phi, alpha)
    return w, h, iters_to_converge, E_final


def main():
    np.random.seed(42)
    N, M = 100, 200
    eta = 0.0005
    p0 = 0.5
    alpha = 1.0
    
    rho_values = [0.0, 0.0005, 0.001, 0.002, 0.003]
    
    activations = {
        'identity': (identity, identity_grad),
        'tanh': (tanh_act, tanh_grad),
        'relu': (relu, relu_grad),
    }
    
    # Generate data from identity (linear) network
    rng = np.random.default_rng(42)
    w0_true = rng.standard_normal(N)
    h0_true = (rng.random(N) < p0).astype(float)
    X = rng.standard_normal((M, N))
    y = X @ (w0_true * h0_true)
    
    os.makedirs("results", exist_ok=True)
    rows = []
    
    for act_name, (phi, phi_grad_fn) in activations.items():
        print(f"\nActivation: {act_name}")
        for rho in rho_values:
            rng2 = np.random.default_rng(int(rho * 10000) + 1)
            w_init = rng2.standard_normal(N) * 0.5
            h_init = np.ones(N)
            
            w_final, h_final, iters, E_final = glauber_activation(
                w_init, h_init, X, y, eta, rho, phi, phi_grad_fn,
                T=30, alpha=alpha, rng=rng2
            )
            
            hd = hamming_distance(h_final.astype(int), h0_true.astype(int))
            print(f"  rho={rho:.4f}: Hamming={hd:.3f} iters={iters} E={E_final:.4f}")
            rows.append((act_name, rho, hd, iters, E_final))
    
    with open("results/activation_comparison.csv", "w") as f:
        f.write("activation,rho,Hamming,iterations,energy\n")
        for act_name, rho, hd, iters, E in rows:
            f.write(f"{act_name},{rho},{hd:.4f},{iters},{E:.6f}\n")
    
    print("\nSaved results/activation_comparison.csv")


if __name__ == "__main__":
    main()
