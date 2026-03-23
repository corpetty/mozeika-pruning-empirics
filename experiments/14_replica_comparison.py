"""
Multi-replica Glauber dynamics for Mozeika pruning.
Minimal version with only key n values for quick execution.
"""
import numpy as np
import os

from pruning_core.replicas import MultiReplicaGlauber


def run_replica_sweep():
    """
    Run with key n values and fixed rho for quick validation.
    """
    # Use few values for quick execution
    n_list = [1, 2, 4, 8]
    rho = 0.001  # Fixed rho
    M = 30  # Small
    d = 30  # Small
    T = 10
    n_seeds = 2
    
    results = []
    
    print(f"Running with n in {n_list}, rho={rho}, M={M}, d={d}, T={T}\n")
    
    for n in n_list:
        print(f"n = {n}")
        replica = MultiReplicaGlauber(n_replicas=n, eta_val=0.0001, alpha=1.0)
        
        hamming_seeds = []
        for seed in range(n_seeds):
            seed_rng = np.random.RandomState(1000 + seed)
            X = seed_rng.randn(M, d)
            h_true = (seed_rng.rand(d) < 0.3).astype(float)
            w_true = seed_rng.randn(d)
            y = X @ (w_true * h_true) + seed_rng.randn(M) * 0.01
            
            w_init = seed_rng.randn(d, 1) * 0.1
            h_init = np.ones((d, 1), dtype=float)
            w_chains = [[w_init] for _ in range(n)]
            
            w_final_chains, h_final, losses = replica.run(
                w_chains, [h_init], X, y, [0.0001], [rho], [1.0],
                T=T, T_h=1.0, rng=np.random.default_rng(2024 + seed)
            )
            
            h_final_flat = h_final[0].flatten()
            hamming = np.mean((h_final_flat - h_true) ** 2)
            hamming_seeds.append(hamming)
            print(f"  seed={seed}: hamming={hamming:.4f}")
        
        mean_h = np.mean(hamming_seeds)
        std_h = np.std(hamming_seeds)
        print(f"  n={n}: mean={mean_h:.4f} +/- {std_h:.4f}\n")
        results.append({'n': n, 'rho': rho, 'hamming_mean': mean_h, 'hamming_std': std_h})
    
    return results


def main():
    os.makedirs('results', exist_ok=True)
    results = run_replica_sweep()
    
    csv_path = 'results/replica_comparison.csv'
    with open(csv_path, 'w') as f:
        f.write('n,rho,hamming_mean,hamming_std\n')
        for r in results:
            f.write(f"{r['n']},{r['rho']:.6f},{r['hamming_mean']:.6f},{r['hamming_std']:.6f}\n")
    
    print(f"\nSaved to {csv_path}")
    print("\n=== Summary ===")
    for n in sorted(set(r['n'] for r in results)):
        print(f"n={n}: mean_hamming={np.mean([r['hamming_mean'] for r in results if r['n']==n]):.4f}")


if __name__ == "__main__":
    main()
