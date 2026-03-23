"""
Phase Diagram Experiment: Find critical rho where phase transition occurs.

Sweeps rho to find the critical point where Hamming(h_final, h0) drops from ~0.5 to ~0.01.
This validates the theoretical prediction of a phase transition in the Ising-perceptron model.

For efficiency, uses N=200, M=400, T=50, K_adam=20 instead of N=500.

Output: results/phase_diagram.csv with columns: eta, rho, Hamming, rho_c_estimate
"""
import numpy as np
from scipy.optimize import curve_fit
import os

from pruning_core import (
    sample_perceptron, total_energy, squared_loss, double_well,
    run_glauber, hamming_distance, sparsity_ratio,
    fast_learning, optimize_w
)


def sigmoid(rho, rho_c, k):
    """Sigmoid model for phase transition: f(rho) = 0.5 * (1 - tanh(k * (rho - rho_c)))"""
    return 0.5 * (1 - np.tanh(k * (rho - rho_c) * 1000))


def fit_sigmoid(rho_values, hamming_values):
    """Fit sigmoid to get critical rho_c."""
    try:
        # Initial guess: rho_c at median rho, k=100 for steep transition
        p0 = [np.median(rho_values), 10.0]
        popt, _ = curve_fit(sigmoid, rho_values, hamming_values, p0=p0, maxfev=5000)
        return popt[0]  # return rho_c
    except Exception as e:
        print(f"  Warning: sigmoid fit failed: {e}")
        return np.median(rho_values)  # fallback


def main():
    np.random.seed(42)
    
    # Parameters (smaller for speed)
    N = 200  # number of parameters  
    M = 400  # training samples
    T = 50   # max iterations
    K_adam = 20  # Adam steps per sweep
    
    # Grid parameters
    eta_set = np.array([0.0001, 0.0005, 0.001])
    rho_set = np.linspace(0, 0.002, 25)
    
    # Generate data
    print(f"Generating data: N={N}, M={M}...")
    X, y, w0, h0 = sample_perceptron(N, M, p0=0.5, sigma=0.01, seed=9900)
    p0_true = sparsity_ratio(h0)
    print(f"True sparsity: {p0_true:.2%}")
    print()
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Results storage
    results = []
    transitions = []
    
    # Run grid search
    for ei, eta in enumerate(eta_set):
        print(f"eta = {eta:.6f}")
        
        for ri, rho in enumerate(rho_set):
            # Run Glauber dynamics
            result = run_glauber(
                np.random.randn(N),
                np.ones(N),
                X, y, eta, rho, alpha=1.0,
                T=T, rng=np.random.default_rng(42 + ei * 10000 + ri * 100)
            )
            
            w_final, h_final = result['w'], result['h']
            
            # Compute Hamming distance to ground truth (normalized)
            Hamming = hamming_distance(h_final.astype(int), h0.astype(int)) / N
            
            # Also track final energy
            E_final = total_energy(w_final, h_final, X, y, eta, 1.0, rho)
            
            print(f"  rho={rho:.6f}: Hamming={Hamming:.4f}, E={E_final:.4f}, sparsity={sparsity_ratio(h_final):.2%}")
            
            results.append({
                'eta': eta,
                'rho': rho,
                'Hamming': Hamming,
                'E': E_final,
                'sparsity': sparsity_ratio(h_final)
            })
        
        # Fit sigmoid for this eta
        rho_vals = np.array([r['rho'] for r in results if abs(r['eta'] - eta) < 1e-8])
        ham_vals = np.array([r['Hamming'] for r in results if abs(r['eta'] - eta) < 1e-8])
        rho_c = fit_sigmoid(rho_vals, ham_vals)
        transitions.append({'eta': eta, 'rho_c': rho_c})
        print(f"  -> rho_c for eta={eta:.6f}: {rho_c:.6f}")
        print()
    
    # Write results to CSV
    csv_path = "results/phase_diagram.csv"
    with open(csv_path, 'w') as f:
        f.write("eta,rho,Hamming,rho_c_estimate\n")
        for r in results:
            # Find appropriate rho_c for this eta
            rho_c = [t['rho_c'] for t in transitions if abs(t['eta'] - r['eta']) < 1e-8][0]
            f.write(f"{r['eta']},{r['rho']},{r['Hamming']},{rho_c}\n")
    
    print(f"\nResults saved to {csv_path}")
    print(f"Critical rho_c values:")
    for t in transitions:
        print(f"  eta={t['eta']:.6f}: rho_c={t['rho_c']:.6f}")
    
    return results


if __name__ == "__main__":
    main()
