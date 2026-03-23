"""
Reproduce the 11x11 grid experiment from the R implementation.

This replicates perceptron_pruning_v5.1.r in Python.

Parameters:
    N=500, p0=0.5, M=1000, M_test=1000, T=100, seed=9900

Output: CSV with columns matching 9900_stats.csv
"""
import numpy as np
from pruning_core import (
    squared_loss, total_energy, double_well,
    run_glauber, exhaustive_search,
    hamming_distance, mse_w, sparsity_ratio,
    optimize_w
)
from pruning_core.data import sample_perceptron


def main():
    # Parameters matching R code
    seed = 9900
    np.random.seed(seed)
    
    N = 500  # number of parameters
    p0 = 0.5  # target sparsity
    M = 1000  # training samples
    M_test = 1000  # test samples
    T = 100  # max iterations
    
    # Grid parameters
    eta_set = np.linspace(0, 0.001, 11)
    rho_set = np.linspace(0, 0.001, 11)
    
    # Generate data (fixed for all experiments)
    print(f"Generating data with seed={seed}...")
    X, y, w0, h0 = sample_perceptron(N, M, p0, sigma=0.01, seed=seed)
    
    # Generate test data
    rng_test = np.random.default_rng(seed + 1)
    X_test = rng_test.standard_normal((M_test, N)) / np.sqrt(N)
    y_test = X_test @ (w0 * h0) + 0.01 * rng_test.standard_normal(M_test)
    
    # Compute true loss (baseline)
    L_true = squared_loss(w0 * h0, h0, X, y, phi=None)
    print(f"True loss (baseline): {L_true:.6f}")
    print(f"True sparsity: {1 - sparsity_ratio(h0):.2%}")
    print()
    
    # Output CSV file
    out_file = f"{seed}_stats.csv"
    
    # Header
    with open(out_file, 'w') as f:
        f.write("eta\trho\tit\tsum_h\tHamming\tMSE\tE\ttrain_error\ttest_error\n")
    
    # Store train/test errors for later analysis
    train_errors = np.zeros((len(eta_set), len(rho_set)))
    test_errors = np.zeros((len(eta_set), len(rho_set)))
    
    # Run grid search
    ROW = 1
    for ri, rho in enumerate(rho_set):
        for ei, eta in enumerate(eta_set):
            print(f"Running: eta={eta:.6f}, rho={rho:.6f}")
            
            # Storage
            E_h = np.zeros(T)
            
            # Initial weights
            w1 = np.random.randn(N)
            
            # Initial optimization with full mask
            w1 = optimize_w(w1, np.ones(N), X, y, eta, K=100, lr=1e-2)
            h1 = np.ones(N)
            
            E_diff = 1.0
            it = 1
            rng_exp = np.random.default_rng(seed + 1000 + ei * 100 + ri)
            
            # Glauber dynamics loop
            while E_diff > 0 and it <= T:
                # One full sweep over all coordinates in random order
                for i in range(N):
                    j = rng_exp.integers(0, N)
                    
                    h2 = h1.copy()
                    h2[j] = 1 - h2[j]  # Flip h[j]
                    
                    # Optimize w for this mask
                    w2 = optimize_w(w1, h2, X, y, eta, K=20, lr=1e-2)
                    
                    # Compute energy difference
                    # Note: R code adds 0.5*rho*(h2[j] - h1[j]) separately
                    # But we include rho in total_energy, so we need to handle this
                    L1 = squared_loss(w1 * h1, h1, X, y, phi=None)
                    L2 = squared_loss(w2 * h2, h2, X, y, phi=None)
                    reg1 = (eta / 2) * np.sum(w1 ** 2)
                    reg2 = (eta / 2) * np.sum(w2 ** 2)
                    V1 = double_well(h1, 1.0)  # alpha=1.0
                    V2 = double_well(h2, 1.0)
                    
                    # Delta E = E(w2, h2) - E(w1, h1) with rho added separately
                    delta = (L2 + reg2 + V2 + 0.5 * rho * h2[j]) - \
                            (L1 + reg1 + V1 + 0.5 * rho * h1[j])
                    
                    if delta < 0:
                        h1 = h2
                        w1 = w2
                
                # Store energy
                E_current = E_h[it - 1] = total_energy(w1, h1, X, y, eta, 1.0, rho)
                
                if it > 1:
                    E_diff = E_h[it - 2] - E_h[it - 1]
                
                # Print progress
                Hamming = np.sum((h1 - h0) ** 2)
                MSE = np.sum((w1 * h1 - w0 * h0) ** 2) / N
                print(f"  it={it}, E={E_current:.4f}, ||h-h0||^2={Hamming:.2f}, "
                      f"Hamming/d={Hamming/N:.4f}, MSE={MSE:.6f}")
                
                it += 1
            
            print()
            
            # Compute metrics
            H_iter = it - 1  # Final iteration count
            sum_h = int(np.sum(h1))
            Hamming_norm = np.sum((h1 - h0) ** 2) / N
            MSE_w = np.sum((w1 * h1 - w0 * h0) ** 2) / N
            E_final = total_energy(w1, h1, X, y, eta, 1.0, rho)
            
            # Train error
            train_error = squared_loss(w1 * h1, h1, X, y, phi=None) / M
            
            # Test error
            L_test_val = squared_loss(w1 * h1, h1, X_test, y_test, phi=None)
            test_error = L_test_val / M_test
            
            # Write to CSV
            with open(out_file, 'a') as f:
                f.write(f"{eta}\t{rho}\t{H_iter}\t{sum_h}\t{Hamming_norm}\t"
                       f"{MSE_w}\t{E_final}\t{train_error}\t{test_error}\n")
            
            train_errors[ei, ri] = train_error
            test_errors[ei, ri] = test_error
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total experiments: {len(eta_set) * len(rho_set)}")
    print(f"eta grid: [{eta_set[0]:.6f}, ..., {eta_set[-1]:.6f}]")
    print(f"rho grid: [{rho_set[0]:.6f}, ..., {rho_set[-1]:.6f}]")
    print(f"Output file: {out_file}")
    print()
    
    # Find critical rho for phase transition
    print("Phase transition analysis:")
    print("-" * 60)
    with open(out_file, 'r') as f:
        lines = f.readlines()[1:]
        for ei, eta in enumerate(eta_set):
            # Find line for this eta
            found = False
            for line in lines:
                l_eta = float(line.split()[0])
                if abs(l_eta - eta) < 1e-10:
                    Hamming = float(line.split()[4])
                    rho = float(line.split()[1])
                    if Hamming <= 0.1:
                        if not found:
                            print(f"  eta={eta:.6f}: found Hamming<=0.1 at rho={rho:.6f}")
                            found = True
                        break
    
    return out_file


if __name__ == "__main__":
    main()
