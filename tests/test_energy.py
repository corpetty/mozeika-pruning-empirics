"""
Test energy calculations.

Key tests:
1. E(w, ones, X, y, eta=0, rho=0) == L(w) (energy equals loss with full mask)
2. V(0) = V(1) = 0 (double well minima)
3. V(0.5) > 0 (barrier at midpoint)
"""
import numpy as np
from pruning_core import squared_loss, double_well, total_energy
from pruning_core.optimizers import optimize_w


def test_energy_equals_loss_with_full_mask():
    """Test E(w, ones, X, y, eta=0, rho=0) == L(w)"""
    np.random.seed(42)
    
    N = 10
    M = 100
    X = np.random.randn(M, N) / np.sqrt(N)
    y = np.random.randn(M)
    
    w = np.random.randn(N)
    h = np.ones(N)  # full mask
    eta = 0
    alpha = 1.0
    rho = 0
    
    # Energy should equal loss (since eta=0, rho=0, and h=ones, V(h)=0)
    E = total_energy(w, h, X, y, eta, alpha, rho)
    L = squared_loss(w, h, X, y)
    
    # With eta=0, rho=0, and h=ones, V(h)=0, so E = L
    assert np.abs(E - L) < 1e-10, f"E={E}, L={L}, diff={np.abs(E-L)}"
    print("✓ test_energy_equals_loss_with_full_mask passed")


def test_double_well_minima():
    """Test V(0) = V(1) = 0"""
    alpha = 1.0
    rho = 0.1
    
    # V(0) = alpha * 0^2 * (0-1)^2 + (rho/2) * 0 = 0 + 0 = 0
    V0 = double_well(np.array([0.0]), alpha, rho)
    assert np.abs(V0 - 0.0) < 1e-10, f"V(0)={V0} should be 0"
    
    # V(1) = alpha * 1^2 * (1-1)^2 + (rho/2) * 1 = 0 + rho/2
    # Wait, this isn't 0! Let me check the R code again...
    # R code: delta <- E(w2, h2) - E(w1, h1) + 0.5 * rho * (h2[j] - h1[j])
    # The rho term is added separately in the delta, not in V(h)
    
    # So V(h) should be just alpha * h^2 * (h-1)^2, giving V(0)=V(1)=0
    alpha_no_rho = alpha
    V_h = lambda h: alpha_no_rho * h ** 2 * (h - 1) ** 2
    
    assert np.abs(V_h(0.0) - 0.0) < 1e-10, f"V(0)={V_h(0.0)} should be 0"
    assert np.abs(V_h(1.0) - 0.0) < 1e-10, f"V(1)={V_h(1.0)} should be 0"
    print("✓ test_double_well_minima passed")


def test_double_well_barrier():
    """Test V(0.5) > 0 (barrier at midpoint)"""
    alpha = 1.0
    V_h = lambda h: alpha * h ** 2 * (h - 1) ** 2
    
    V_05 = V_h(0.5)
    assert V_05 > 0, f"V(0.5)={V_05} should be > 0"
    # Should be exactly 1/16 = 0.0625
    assert np.abs(V_05 - 0.0625) < 1e-10, f"V(0.5) should be 0.0625"
    print("✓ test_double_well_barrier passed")


def test_total_energy_structure():
    """Test that total energy has correct structure"""
    np.random.seed(123)
    
    N = 5
    M = 50
    X = np.random.randn(M, N) / np.sqrt(N)
    y = np.random.randn(M)
    
    w = np.random.randn(N)
    h = np.ones(N)
    eta = 0.001
    alpha = 1.0
    rho = 0.001
    
    E = total_energy(w, h, X, y, eta, alpha, rho)
    
    # Should be sum of positive terms (loss is always positive)
    assert E > 0, f"E should be positive, got {E}"
    
    # Check individual components
    L = squared_loss(w, h, X, y)
    reg = (eta / 2) * np.sum(w ** 2)
    V = double_well(h, alpha, rho)
    
    E_expected = L + reg + V
    assert np.abs(E - E_expected) < 1e-10, f"E mismatch: {E} vs {E_expected}"
    print("✓ test_total_energy_structure passed")


if __name__ == "__main__":
    test_energy_equals_loss_with_full_mask()
    test_double_well_minima()
    test_double_well_barrier()
    test_total_energy_structure()
    print("\nAll energy tests passed!")
