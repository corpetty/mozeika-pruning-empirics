"""
Test dynamics module.

Key test: Glauber on tiny N=5 example, energy is non-increasing
"""
import numpy as np
from pruning_core.dynamics import Glauber, run_glauber, exhaustive_search
from pruning_core.energy import total_energy


def test_glauber_energy_non_increasing():
    """Test Glauber dynamics energy is non-increasing"""
    np.random.seed(42)
    
    N = 5
    M = 50
    X = np.random.randn(M, N) / np.sqrt(N)
    y = np.random.randn(M)
    
    # Small regularization and potential
    eta = 0.01
    rho = 0.001
    alpha = 1.0
    
    w_init = np.random.randn(N)
    h_init = np.ones(N)
    
    # Run a few Glauber steps and check energy
    rng = np.random.default_rng(123)
    
    w = w_init.copy()
    h = h_init.copy()
    energies = []
    
    for _ in range(5):
        # One full sweep
        h, w, flips = Glauber.step(w, h, X, y, eta, rho, alpha, rng)
        E = total_energy(w, h, X, y, eta, alpha, rho)
        energies.append(E)
    
    # Energy should be non-increasing (or same)
    for i in range(1, len(energies)):
        assert energies[i] <= energies[i-1] + 1e-10, \
            f"Energy increased: {energies[i-1]} -> {energies[i]}"
    
    print(f"✓ test_glauber_energy_non_increasing passed (energies: {energies})")


def test_exhaustive_search_tiny():
    """Test exhaustive search on tiny example"""
    np.random.seed(123)
    
    N = 4
    M = 20
    X = np.random.randn(M, N) / np.sqrt(N)
    y = np.random.randn(M)
    
    eta = 0.01
    rho = 0.001
    alpha = 1.0
    
    # Exhaustive search should find the best mask
    result = exhaustive_search(X, y, eta, rho, alpha, N, K_adam=10)
    
    # Check that we got a valid result
    assert result['w'] is not None, "Should return non-None w"
    assert result['h'] is not None, "Should return non-None h"
    assert result['E'] is not None, "Should return E"
    assert len(result['h']) == N, "h should have length N"
    
    # Verify the returned energy matches
    E_verify = total_energy(result['w'], result['h'], X, y, eta, alpha, rho)
    assert np.abs(E_verify - result['E']) < 1e-8, f"E mismatch: {E_verify} vs {result['E']}"
    
    print(f"✓ test_exhaustive_search_tiny passed (E={result['E']:.4f})")


def test_glauber_vs_exhaustive():
    """Test that Glauber and exhaustive search agree on very small problem"""
    np.random.seed(456)
    
    N = 3  # Very small for 2^N=8 masks
    M = 30
    X = np.random.randn(M, N) / np.sqrt(N)
    y = np.random.randn(M)
    
    eta = 0.005
    rho = 0.0005
    alpha = 0.5
    
    # Run Glauber multiple times with different seeds to find best
    best_glauber_E = float('inf')
    rng_base = np.random.default_rng(789)
    
    for _ in range(10):
        rng = rng_base.choice([0, 1, 2, 3, 4])
        run_rng = np.random.default_rng(rng)
        result = run_glauber(
            np.random.randn(N), 
            np.ones(N), 
            X, y, eta, rho, alpha, 
            T=50, 
            rng=run_rng
        )
        # Note: run_glauber returns the final E via total_energy, not in result dict
        E_glauber = total_energy(result['w'], result['h'], X, y, eta, alpha, rho)
        if E_glauber < best_glauber_E:
            best_glauber_E = E_glauber
            best_glauber_h = result['h'].copy()
    
    # Run exhaustive search
    result_ex = exhaustive_search(X, y, eta, rho, alpha, N, K_adam=20)
    
    # Check that exhaustive search found better or equal energy
    # (Glauber might not find global optimum due to local minima)
    assert result_ex['E'] <= best_glauber_E + 0.1, \
        f"Exhaustive search should find better or equal energy: {result_ex['E']} vs {best_glauber_E}"
    
    print(f"✓ test_glauber_vs_exhaustive passed (exhaustive E={result_ex['E']:.4f}, best Glauber E={best_glauber_E:.4f})")


if __name__ == "__main__":
    test_glauber_energy_non_increasing()
    test_exhaustive_search_tiny()
    test_glauber_vs_exhaustive()
    print("\nAll dynamics tests passed!")
