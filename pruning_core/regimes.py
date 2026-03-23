"""
Dynamical regimes for the Ising-perceptron model.

Based on Mozeika & Pizzoferrato paper, four regimes based on timescale ratio:
1. Equal timescales — joint Langevin on w and h
2. Fast learning (tau_w << tau_h) — Adam inner loop + Glauber outer
3. Fast pruning (tau_h << tau_w) — masks update fast, weights lag
4. Low temperature (beta->inf) — MAP limit (default implementation)
"""
import numpy as np
from .energy import total_energy, squared_loss, double_well
from .optimizers import optimize_w
from .dynamics import Glauber


def joint_langevin(w_init, h_init, X, y, eta, rho, alpha, T_w=0.1, T_h=0.1, alpha_rw=1e-3, alpha_rh=1e-3, T=100, rng=None):
    """
    Equal timescales: simultaneous stochastic updates on w and h.
    
    Implements joint Langevin dynamics where both weights and masks
    undergo continuous-time stochastic evolution:
    
    w update: w += -tau_w * grad_E_w + sqrt(2*T_w*tau_w) * noise_w
    h update: for each j, flip h[j] with prob exp(-beta_h * max(0, delta_E))
    
    Args:
        w_init: initial weights
        h_init: initial mask (all ones typically)
        X: inputs (M, N)
        y: targets (M,)
        eta: L2 regularization coefficient
        rho: sparsity pressure
        alpha: double-well barrier coefficient
        T_w: temperature for weight dynamics
        T_h: temperature for mask dynamics (inverse beta)
        alpha_rw: learning rate for w (timescale)
        alpha_rh: learning rate for h (timescale)
        T: maximum iterations/sweeps
        rng: random number generator
    
    Returns:
        dict with 'w', 'h', 'losses', 'history', 'iterations'
    """
    if rng is None:
        rng = np.random.default_rng()
    
    N = len(h_init)
    w = w_init.copy().astype(float)
    h = h_init.copy().astype(float)
    
    # Initial optimization
    w = optimize_w(w, h, X, y, eta, K=10, lr=alpha_rw)
    
    losses = []
    history = {'w': [w.copy()], 'h': [h.copy()]}
    
    for it in range(T):
        # Weight update: Langevin step
        # grad_E_w = X^T (X w) + eta * w (for squared loss + L2)
        loss_grad = squared_loss(w * h, h, X, y)
        grad_loss = (X.T @ (X @ w)) / len(y)  # simplified gradient
        grad_regularization = eta * w
        grad_w = grad_loss + grad_regularization
        
        # Langevin update: w = w - alpha_rw * grad + sqrt(2 * T_w * alpha_rw) * noise
        noise_w = rng.standard_normal(N)
        w = w - alpha_rw * grad_w + np.sqrt(2 * T_w * alpha_rw) * noise_w
        
        # Mask update: Glauber sweep with T_h temperature
        # For each coordinate, compute acceptance probability
        for j in rng.permutation(N):
            h_try = h.copy()
            h_try[j] = 1 - h_try[j]
            
            E_current = total_energy(w, h, X, y, eta, alpha, rho)
            E_try = total_energy(w, h_try, X, y, eta, alpha, rho)
            delta_E = E_try - E_current
            
            # Glauber acceptance: accept if delta_E < 0, else accept with prob exp(-delta_E/T_h)
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T_h):
                h = h_try
        
        # Store energy
        E_current = total_energy(w, h, X, y, eta, alpha, rho)
        losses.append(E_current)
        history['w'].append(w.copy())
        history['h'].append(h.copy())
    
    return {
        'w': w,
        'h': h,
        'losses': losses,
        'history': history,
        'iterations': len(losses)
    }


def fast_pruning(w_init, h_init, X, y, eta, rho, alpha, K_w=5, T=100, T_h=1.0, rng=None):
    """
    Fast pruning (tau_h << tau_w): masks update fast, weights lag.
    
    Implements regime where h updates many times per w update:
    - Inner loop: K_w Glauber sweeps on h (using current w)
    - Outer loop: single Adam step on w, then many Glauber steps
    
    This is the "fast pruning" regime where the mask tries to
    find optimal sparsity pattern while weights slowly adapt.
    
    Args:
        w_init: initial weights
        h_init: initial mask
        X: inputs (M, N)
        y: targets (M,)
        eta: L2 regularization
        rho: sparsity pressure
        alpha: double-well barrier
        K_w: number of Glauber sweeps per Adam step (default: 5, small means h updates faster)
        T: maximum outer iterations
        T_h: temperature for Glauber (default: 1.0)
        rng: random number generator
    
    Returns:
        dict with 'w', 'h', 'losses', 'history', 'iterations'
    """
    if rng is None:
        rng = np.random.default_rng()
    
    w = w_init.copy().astype(float)
    h = h_init.copy().astype(float)
    
    # Initial optimization
    w = optimize_w(w, h, X, y, eta, K=10, lr=1e-2)
    
    losses = []
    history = {'w': [w.copy()], 'h': [h.copy()]}
    
    for it in range(T):
        # Inner loop: K_w Glauber sweeps
        for _ in range(K_w):
            h, _, _ = Glauber.step(w, h, X, y, eta, rho, alpha, rng)
        
        # Outer loop: single Adam step on w
        w = optimize_w(w, h, X, y, eta, K=1, lr=1e-2)
        
        # Store energy
        E_current = total_energy(w, h, X, y, eta, alpha, rho)
        losses.append(E_current)
        history['w'].append(w.copy())
        history['h'].append(h.copy())
    
    return {
        'w': w,
        'h': h,
        'losses': losses,
        'history': history,
        'iterations': len(losses)
    }


def fast_learning(w_init, h_init, X, y, eta, rho, alpha, K_adam=20, T=100, rng=None):
    """
    Fast learning (tau_w << tau_h): Adam inner loop + Glauber outer.
    
    This is the current implementation: many Adam steps per one Glauber sweep.
    Weights adjust quickly to current mask, then mask updates slowly.
    
    Args:
        w_init: initial weights
        h_init: initial mask
        X: inputs (M, N)
        y: targets (M,)
        eta: L2 regularization
        rho: sparsity pressure
        alpha: double-well barrier
        K_adam: Adam steps per Glauber sweep (default: 20)
        T: maximum iterations
        rng: random number generator
    
    Returns:
        dict with 'w', 'h', 'losses', 'history', 'iterations'
    """
    if rng is None:
        rng = np.random.default_rng()
    
    w = w_init.copy().astype(float)
    h = h_init.copy().astype(float)
    
    # Initial optimization
    w = optimize_w(w, h, X, y, eta, K=10, lr=1e-2)
    
    losses = []
    history = {'w': [w.copy()], 'h': [h.copy()]}
    
    for it in range(T):
        # Inner loop: K_adam Adam steps (weights equilibrate)
        w = optimize_w(w, h, X, y, eta, K=K_adam, lr=1e-2)
        
        # Outer loop: one Glauber sweep
        h, _, _ = Glauber.step(w, h, X, y, eta, rho, alpha, rng)
        
        # Store energy
        E_current = total_energy(w, h, X, y, eta, alpha, rho)
        losses.append(E_current)
        history['w'].append(w.copy())
        history['h'].append(h.copy())
    
    return {
        'w': w,
        'h': h,
        'losses': losses,
        'history': history,
        'iterations': len(losses)
    }
