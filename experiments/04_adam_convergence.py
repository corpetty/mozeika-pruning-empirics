"""
Adam Convergence Test: How many Adam steps is "enough"?

Tests the assumption that w is fully equilibrated before h updates.
For a fixed (eta, rho) in the recovery region, sweeps K in {5, 10, 20, 50, 100}.

Output: results/adam_convergence.csv with columns: K, Hamming, iterations
"""
import numpy as np
import os

from pruning_core import (
    sample_perceptron, total_energy,
    run_glauber, hamming_distance, sparsity_ratio,
    optimize_w
)


class AdaptedGlauber:
    """
    Modified Glauber to expose intermediate results for analysis.
    """
    
    @staticmethod
    def step_with_adam(w, h, X, y, eta, rho, alpha, K_adam, rng=None):
        """
        Run K_adam Adam steps on w, then one Glauber sweep on h.
        
        Returns: w_after_adam, h_after_glauber, flips_accepted
        """
        if rng is None:
            rng = np.random.default_rng()
        
        N = len(h)
        
        # Adam steps on w
        w_new = optimize_w(w, h, X, y, eta, K=K_adam, lr=1e-2)
        
        # Glauber sweep on h
        h_new = h.copy()
        flips = 0
        
        order = rng.permutation(N)
        for j in order:
            h_try = h_new.copy()
            h_try[j] = 1 - h_try[j]
            
            w_try = optimize_w(w_new, h_try, X, y, eta, K=20, lr=1e-2)
            
            E_current = total_energy(w_new, h_new, X, y, eta, alpha, rho)
            E_try = total_energy(w_try, h_try, X, y, eta, alpha, rho)
            
            delta = E_try - E_current
            
            if delta < 0:
                h_new = h_try
                w_new = w_try
                flips += 1
        
        return w_new, h_new, flips


def run_adam_glauber(w_init, h_init, X, y, eta, rho, alpha, K_adam, T=100, rng=None):
    """
    Run the modified dynamics: K_adam Adam steps then Glauber sweep, T times.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    w = w_init.copy()
    h = h_init.copy()
    
    # Initial optimization
    w = optimize_w(w, h, X, y, eta, K=100, lr=1e-2)
    
    losses = []
    history = {'w': [w.copy()], 'h': [h.copy()]}
    
    for it in range(T):
        w, h, flips = AdaptedGlauber.step_with_adam(
            w, h, X, y, eta, rho, alpha, K_adam, rng
        )
        
        E = total_energy(w, h, X, y, eta, alpha, rho)
        losses.append(E)
        history['w'].append(w.copy())
        history['h'].append(h.copy())
    
    return {
        'w': w,
        'h': h,
        'losses': losses,
        'history': history,
        'iterations': len(losses)
    }


def main():
    np.random.seed(42)
    
    # Parameters
    N = 200
    M = 400
    T = 50
    alpha = 1.0
    eta = 0.0005  # fixed
    rho = 0.0005  # fixed - in recovery region
    
    # K values to sweep
    K_values = [5, 10, 20, 50, 100]
    
    # Generate data
    print(f"Generating data: N={N}, M={M}...")
    X, y, w0, h0 = sample_perceptron(N, M, p0=0.5, sigma=0.01, seed=9900)
    print(f"True sparsity: {sparsity_ratio(h0):.2%}")
    print()
    
    os.makedirs("results", exist_ok=True)
    
    results = []
    
    # Run for each K
    for K in K_values:
        print(f"K = {K} Adam steps per sweep")
        
        Hamming_total = 0
        iterations_total = 0
        
        # Run multiple seeds for averaging
        for seed in range(5):
            rng = np.random.default_rng(42 + seed)
            
            result = run_adam_glauber(
                np.random.randn(N),
                np.ones(N),
                X, y, eta, rho, alpha, K, T, rng
            )
            
            w_final, h_final = result['w'], result['h']
            Hamming = hamming_distance(h_final.astype(int), h0.astype(int)) / N
            iters = result['iterations']
            
            Hamming_total += Hamming
            iterations_total += iters
            
            print(f"  seed={seed}: Hamming={Hamming:.4f}, iters={iters}")
        
        Hamming_avg = Hamming_total / 5
        iterations_avg = iterations_total / 5
        
        print(f"  -> Average: Hamming={Hamming_avg:.4f}, iterations={iterations_avg:.1f}")
        
        results.append({
            'K': K,
            'Hamming': Hamming_avg,
            'iterations': iterations_avg
        })
        print()
    
    # Write results
    csv_path = "results/adam_convergence.csv"
    with open(csv_path, 'w') as f:
        f.write("K,Hamming,iterations\n")
        for r in results:
            f.write(f"{r['K']},{r['Hamming']},{r['iterations']}\n")
    
    print(f"Results saved to {csv_path}")
    
    # Find minimum K where Hamming stabilizes
    print("\nAnalysis:")
    Hamming_changes = [results[i+1]['Hamming'] - results[i]['Hamming'] for i in range(len(results)-1)]
    for i, change in enumerate(Hamming_changes):
        print(f"  K from {results[i]['K']} to {results[i+1]['K']}: delta_Hamming={change:.4f}")
    
    return results


if __name__ == "__main__":
    main()
