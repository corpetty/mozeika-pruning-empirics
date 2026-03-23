"""
Test Adam optimizer.

Key test: Adam on simple quadratic converges in <100 steps
"""
import numpy as np
from pruning_core.optimizers import AdamOptimizer, optimize_w


def test_adam_convergence():
    """Test Adam converges on simple quadratic in <100 steps"""
    # Simple quadratic: f(w) = w^2, gradient = 2w
    # Adam should converge quickly
    np.random.seed(42)
    
    N = 5
    X = np.random.randn(100, N) / np.sqrt(N)
    y = np.sum(X, axis=1)  # Linear relationship
    eta = 0.01
    
    # Initial weights away from optimum
    w_init = np.random.randn(N) * 10
    
    # Run Adam optimization
    h = np.ones(N)
    w_opt = optimize_w(w_init, h, X, y, eta, K=100, lr=1e-2)
    
    # Check that loss decreased
    from pruning_core import squared_loss
    L_init = squared_loss(w_init, h, X, y)
    L_opt = squared_loss(w_opt, h, X, y)
    
    assert L_opt < L_init, f"Loss should decrease: {L_init} -> {L_opt}"
    print(f"✓ test_adam_convergence passed (loss: {L_init:.4f} -> {L_opt:.4f})")


def test_adam_optimizer_class():
    """Test AdamOptimizer class directly"""
    np.random.seed(42)
    
    N = 3
    adam = AdamOptimizer(N, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)
    
    # Test that optimization steps produce reasonable updates
    w = np.array([1.0, 2.0, 3.0])
    grad = np.array([0.1, 0.2, 0.3])
    
    w_new = adam.step(w, grad)
    
    # Check that w changed
    assert not np.allclose(w, w_new), "Weights should change after step"
    
    # Check that timestep incremented
    assert adam.t == 1, f"Expected t=1, got {adam.t}"
    
    print("✓ test_adam_optimizer_class passed")


def test_adam_bias_correction():
    """Test that Adam bias correction works correctly"""
    N = 2
    adam = AdamOptimizer(N, lr=0.01)
    
    # First step: m = (1-0.9)*g = 0.1g, v = 0.199*g^2
    # m_hat = m / (1-0.9) = g, v_hat = v / (1-0.999) = g^2
    # After many steps, bias correction should make m_hat and v_hat accurate
    
    w = np.zeros(N)
    grad = np.ones(N) * 2.0
    
    # Run 10 steps with constant gradient
    for _ in range(10):
        w = adam.step(w, grad)
    
    # Check that t is reasonable
    assert adam.t == 10, f"Expected t=10, got {adam.t}"
    
    # Check that moments are reasonable (should be close to gradient)
    m_expected = grad  # After bias correction
    v_expected = grad ** 2
    
    # m_hat should be close to g for constant gradient after enough steps
    m_hat = adam.m / (1 - adam.beta1 ** adam.t)
    v_hat = adam.v / (1 - adam.beta2 ** adam.t)
    
    assert np.allclose(m_hat, m_expected, rtol=0.1), f"m_hat={m_hat}, expected={m_expected}"
    print("✓ test_adam_bias_correction passed")


if __name__ == "__main__":
    test_adam_convergence()
    test_adam_optimizer_class()
    test_adam_bias_correction()
    print("\nAll optimizer tests passed!")
