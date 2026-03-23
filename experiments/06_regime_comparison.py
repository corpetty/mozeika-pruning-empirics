"""
Regime Comparison: Compare fast_learning vs fast_pruning vs joint_langevin.

For a single (eta, rho) in the transition region (where Hamming ~0.2):
- Run all three implemented regimes
- Measure: Hamming to ground truth, convergence speed, final energy
- Sweep T_w/T_h for joint_langevin to show interpolation between regimes
"""
import numpy as np
import os

from pruning_core import (
    sample_perceptron, total_energy, hamming_distance, sparsity_ratio,
    joint_langevin, fast_pruning, fast_learning
)


def run_regime_comparison(regime_fn, name, w_init, h_init, X, y, eta, rho, alpha,
                          params, T=100, rng=None):
    """Run a regime and return final metrics."""
    if rng is None:
        rng = np.random.default_rng()
    
    result = regime_fn(
        w_init, h_init, X, y, eta, rho, alpha,
        T=T, rng=rng, **params
    )
    
    w_final = result['w']
    h_final = result['h']
    losses = result['losses']
    
    Hamming = hamming_distance(h_final.astype(int), h0.astype(int)) / N_true
    E_final = total_energy(w_final, h_final, X, y, eta, alpha, rho)
    sparsity = sparsity_ratio(h_final)
    conv_iters = len(losses)
    
    # Compute convergence rate (energy drop per iteration)
    if len(losses) > 1:
        total_drop = losses[0] - losses[-1]
        conv_rate = total_drop / conv_iters
    else:
        conv_rate = 0
    
    return {
        'name': name,
        'Hamming': Hamming,
        'E_final': E_final,
        'sparsity': sparsity,
        'conv_iters': conv_iters,
        'conv_rate': conv_rate,
        'losses': losses
    }


def main():
    global h0, N_true
    
    # Fixed parameters for comparison
    eta = 0.0005
    rho = 0.0008  # in transition region
    alpha = 1.0
    N_regime = 200
    T = 80
    
    np.random.seed(42)
    
    # Generate data
    print(f"Generating data: N={N_regime}, M=200...")
    X, y, w0_true, h0_true = sample_perceptron(N_regime, 200, p0=0.5, sigma=0.01, seed=9900)
    N_true = N_regime
    h0 = h0_true  # for use in run_regime_comparison
    
    print(f"Ground truth sparsity: {sparsity_ratio(h0):.2%}")
    print()
    
    # Initialize weights
    w_init = np.random.randn(N_regime)
    h_init = np.ones(N_regime)
    
    results = []
    
    # Regime 1: Fast Learning (default)
    print("Running FAST LEARNING (fast learning regime)...")
    result = run_regime_comparison(
        fast_learning, "fast_learning",
        w_init.copy(), h_init.copy(), X, y, eta, rho, alpha,
        params={'K_adam': 20},
        T=T, rng=np.random.default_rng(42)
    )
    results.append(result)
    print(f"  -> Hamming={result['Hamming']:.4f}, E={result['E_final']:.4f}, iters={result['conv_iters']}")
    
    # Regime 2: Fast Pruning
    print("Running FAST PRUNING (fast pruning regime)...")
    result = run_regime_comparison(
        fast_pruning, "fast_pruning",
        w_init.copy(), h_init.copy(), X, y, eta, rho, alpha,
        params={'K_w': 5},
        T=T, rng=np.random.default_rng(43)
    )
    results.append(result)
    print(f"  -> Hamming={result['Hamming']:.4f}, E={result['E_final']:.4f}, iters={result['conv_iters']}")
    
    # Regime 3: Joint Langevin (equal timescales) - sweep T_w/T_h
    print("Running JOINT LANGEVIN (equal timescales) for different T_w/T_h ratios...")
    temp = [
        (0.01, 0.01, "1x1"),
        (0.05, 0.05, "5x5"),
        (0.1, 0.1, "10x10"),
    ]
    
    for T_w, T_h, label in temp:
        result = run_regime_comparison(
            joint_langevin, f"joint_langevin_Tw{T_w:.2f}_Th{T_h:.2f}",
            w_init.copy(), h_init.copy(), X, y, eta, rho, alpha,
            params={'T_w': T_w, 'T_h': T_h, 'alpha_rw': 1e-3, 'alpha_rh': 1e-3},
            T=T, rng=np.random.default_rng(44)
        )
        results.append(result)
        print(f"  -> Hamming={result['Hamming']:.4f}, E={result['E_final']:.4f}, iters={result['conv_iters']}")
        print()
    
    # Write results
    os.makedirs("results", exist_ok=True)
    csv_path = "results/regime_comparison.csv"
    
    with open(csv_path, 'w') as f:
        f.write("regime,Hamming,E_final,sparsity,conv_iters,conv_rate\n")
        for r in results:
            f.write(f"{r['name']},{r['Hamming']},{r['E_final']},{r['sparsity']:.4f},"
                   f"{r['conv_iters']},{r['conv_rate']:.6f}\n")
    
    print(f"Results saved to {csv_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("REGIME COMPARISON SUMMARY")
    print("=" * 60)
    print("Regime              | Hamming  | E_final  | Sparsity | Iterations")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:20s} | {r['Hamming']:7.4f} | {r['E_final']:8.4f} | "
              f"{r['sparsity']:8.2%} | {r['conv_iters']:10d}")
    
    return results


if __name__ == "__main__":
    main()
