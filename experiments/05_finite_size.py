"""
Finite-Size Scaling: How does the phase transition depend on system size?

Fix M/N = 2, sweep N in {50, 100, 200, 500}.
For each N, run rho sweep (20 points) at optimal eta.
Fit sigmoid slope to Hamming(rho) -> measure transition sharpness.

Expected: slope increases with N (thermodynamic limit).
"""
import numpy as np
from scipy.optimize import curve_fit
import os

from pruning_core import (
    sample_perceptron, total_energy,
    run_glauber, hamming_distance, sparsity_ratio,
)


def sigmoid(rho, rho_c, k):
    """Sigmoid model: f(rho) = 0.5 * (1 - tanh(k * (rho - rho_c)))"""
    return 0.5 * (1 - np.tanh(k * (rho - rho_c) * 1000))


def fit_sigmoid(rho_values, hamming_values):
    """Fit sigmoid to get rho_c and slope."""
    try:
        rho_c_guess = np.median(rho_values)
        k_guess = 100
        popt, _ = curve_fit(sigmoid, rho_values, hamming_values, 
                          p0=[rho_c_guess, k_guess], maxfev=5000)
        return popt[0], popt[1]  # (rho_c, k)
    except Exception as e:
        print(f"  Warning: sigmoid fit failed: {e}")
        return np.median(rho_values), 50.0


def main():
    np.random.seed(42)
    
    # Parameters
    ratio = 2  # M/N = 2
    alpha = 1.0
    eta = 0.0005  # optimal eta
    alpha_val = 1.0
    
    # System sizes to sweep
    N_values = [50, 100, 200, 500]
    rho_sweep = np.linspace(0, 0.002, 20)
    
    os.makedirs("results", exist_ok=True)
    
    results = []
    scaling_data = []
    
    for N in N_values:
        M = int(ratio * N)
        print(f"\nN = {N}, M = {M}")
        
        # Generate data
        X, y, w0, h0 = sample_perceptron(N, M, p0=0.5, sigma=0.01, seed=42 + N)
        
        hamming_results = []
        
        # Sweep rho
        for rho in rho_sweep:
            # Run Glauber
            result = run_glauber(
                np.random.randn(N),
                np.ones(N),
                X, y, eta, rho, alpha_val,
                T=100,
                rng=np.random.default_rng(42 + N + int(rho * 1000))
            )
            
            h_final = result['h']
            Hamming = hamming_distance(h_final.astype(int), h0.astype(int)) / N
            
            hamming_results.append({
                'rho': rho,
                'Hamming': Hamming
            })
            
            print(f"  rho={rho:.6f}: Hamming={Hamming:.4f}")
        
        # Fit sigmoid
        rho_vals = np.array([r['rho'] for r in hamming_results])
        ham_vals = np.array([r['Hamming'] for r in hamming_results])
        rho_c, k = fit_sigmoid(rho_vals, hamming_results)
        
        print(f"  rho_c = {rho_c:.6f}, slope k = {k:.2f}")
        
        scaling_data.append({
            'N': N,
            'rho_c': rho_c,
            'k': k,
            'rho_sweep': rho_sweep.tolist(),
            'hamming_vals': hamming_results
        })
        
        # Save intermediate results
        results.append({
            'N': N,
            'rho': rho_sweep,
            'Hamming': hamming_results,
            'slope': k
        })
    
    # Write master CSV
    csv_path = "results/finite_size.csv"
    with open(csv_path, 'w') as f:
        f.write("N,rho,Hamming,slope\n")
        for entry in scaling_data:
            slope = entry['k']
            for i, rho in enumerate(entry['rho_sweep']):
                hamming = entry['hamming_vals'][i]
                f.write(f"{entry['N']},{rho},{hamming},{slope}\n")
    
    # Save detailed scaling data
    import json
    json_path = "results/finite_size_detailed.json"
    with open(json_path, 'w') as f:
        json.dump(scaling_data, f, indent=2)
    
    print(f"\nResults saved to {csv_path} and {json_path}")
    
    # Summary
    print("\nFinite-Size Scaling Summary:")
    print("N    | rho_c    | slope (k)")
    print("-" * 30)
    for entry in scaling_data:
        print(f"{entry['N']:3d} | {entry['rho_c']:.6f} | {entry['k']:6.2f}")
    
    return scaling_data


if __name__ == "__main__":
    main()
