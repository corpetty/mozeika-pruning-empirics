import numpy as np
from .energy import total_energy, squared_loss, double_well
from .optimizers import optimize_w


class Glauber:
    """
    Glauber dynamics for binary mask updates.
    
    Implements coordinate descent with random order updates.
    For each coordinate, flip if energy decreases.
    """
    
    @staticmethod
    def step(w, h, X, y, eta, rho, alpha, rng=None):
        """
        Perform one full sweep over all coordinates in random order.
        
        For each coordinate j:
            - Flip h[j]
            - Compute energy difference
            - If delta < 0, accept the flip (deterministic Glauber at low T)
        
        Args:
            w: current weights
            h: current mask
            X: inputs (M, N)
            y: targets (M,)
            eta: L2 regularization
            rho: sparsity pressure
            alpha: double-well barrier
            rng: random number generator (optional)
        
        Returns:
            new h, new w, number of flips accepted
        """
        if rng is None:
            rng = np.random.default_rng()
        
        N = len(h)
        h_new = h.copy()
        w_new = w.copy()
        flips = 0
        
        # Random order of coordinates
        order = rng.permutation(N)
        
        for j in order:
            # Try flipping h[j]
            h_try = h_new.copy()
            h_try[j] = 1 - h_try[j]
            
            # Optimize w for this mask
            w_try = optimize_w(w_new, h_try, X, y, eta, K=20, lr=1e-2)
            
            # Compute energy difference
            # Note: R code includes the 0.5*rho*h term in the delta
            E_current = total_energy(w_new, h_new, X, y, eta, alpha, rho)
            E_try = total_energy(w_try, h_try, X, y, eta, alpha, rho)
            
            # Double-well potential term for just this coordinate
            # The R code adds: + 0.5 * rho * (h2[j] - h1[j])
            delta = E_try - E_current
            
            if delta < 0:
                h_new = h_try
                w_new = w_try
                flips += 1
        
        return h_new, w_new, flips


def run_glauber(w_init, h_init, X, y, eta, rho, alpha, T=100, rng=None):
    """
    Run Glauber dynamics until convergence or T iterations.
    
    Args:
        w_init: initial weights
        h_init: initial mask (all ones typically)
        X: inputs (M, N)
        y: targets (M,)
        eta: L2 regularization
        rho: sparsity pressure
        alpha: double-well barrier
        T: maximum iterations (default: 100)
        rng: random number generator (optional)
    
    Returns:
        best (w, h, losses, history_dict) where:
            - w, h: final weights and mask
            - losses: energy at each iteration
            - history_dict: dict of intermediate results
    """
    if rng is None:
        rng = np.random.default_rng()
    
    w = w_init.copy()
    h = h_init.copy().astype(float)
    
    # Initial optimization with full mask
    w = optimize_w(w, h, X, y, eta, K=100, lr=1e-2)
    
    losses = []
    history = {
        'w': [w.copy()],
        'h': [h.copy()],
        'flips': []
    }
    
    it = 0
    E_diff = 1.0
    
    while E_diff > 0 and it < T:
        # Run one Glauber sweep
        h, w, flips = Glauber.step(w, h, X, y, eta, rho, alpha, rng)
        
        # Compute current energy
        E_current = total_energy(w, h, X, y, eta, alpha, rho)
        losses.append(E_current)
        
        if it > 0:
            E_diff = losses[-2] - losses[-1]
        
        history['w'].append(w.copy())
        history['h'].append(h.copy())
        history['flips'].append(flips)
        
        it += 1
    
    return {
        'w': w,
        'h': h,
        'losses': losses,
        'history': history,
        'iterations': it
    }


def run_glauber_finite_temp(w_init, h_init, X, y, eta, rho, alpha, T=100, T_h=0.01, rng=None):
    """
    Run Glauber dynamics with finite-temperature (Metropolis) acceptance.

    Same as run_glauber but accept flip with probability min(1, exp(-ΔE/T_h))
    instead of greedy accept. At T_h→0, recovers zero-temp (greedy) behavior.

    Args:
        w_init: initial weights
        h_init: initial mask (all ones typically)
        X: inputs (M, N)
        y: targets (M,)
        eta: L2 regularization
        rho: sparsity pressure
        alpha: double-well barrier
        T: maximum iterations (default: 100)
        T_h: temperature for mask flips (default: 0.01)
        rng: random number generator (optional)

    Returns:
        dict with w, h, losses, history, iterations
    """
    if rng is None:
        rng = np.random.default_rng()

    w = w_init.copy()
    h = h_init.copy().astype(float)
    N = len(h)

    # Initial optimization with full mask
    w = optimize_w(w, h, X, y, eta, K=100, lr=1e-2)

    losses = []
    history = {
        'w': [w.copy()],
        'h': [h.copy()],
        'flips': []
    }

    for it in range(T):
        flips = 0
        order = rng.permutation(N)

        for j in order:
            h_try = h.copy()
            h_try[j] = 1 - h_try[j]

            w_try = optimize_w(w, h_try, X, y, eta, K=20, lr=1e-2)

            E_current = total_energy(w, h, X, y, eta, alpha, rho)
            E_try = total_energy(w_try, h_try, X, y, eta, alpha, rho)

            delta = E_try - E_current
            if delta < 0:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta / T_h)

            if accept:
                h = h_try
                w = w_try
                flips += 1

        E_current = total_energy(w, h, X, y, eta, alpha, rho)
        losses.append(E_current)
        history['w'].append(w.copy())
        history['h'].append(h.copy())
        history['flips'].append(flips)

    return {
        'w': w,
        'h': h,
        'losses': losses,
        'history': history,
        'iterations': T
    }


def exhaustive_search(X, y, eta, rho, alpha, N, K_adam=50):
    """
    Enumerate all 2^N binary masks and find the best.
    
    For small N (≤20), this computes the exact best mask.
    
    Args:
        X: inputs (M, N)
        y: targets (M,)
        eta: L2 regularization
        rho: sparsity pressure
        alpha: double-well barrier
        N: number of parameters
        K_adam: Adam steps for w optimization
    
    Returns:
        best (w, h, E, h_binary) where:
            - w: optimized weights for best mask
            - h: binary mask
            - E: total energy
            - h_binary: h as integers (0, 1)
    """
    best_E = float('inf')
    best_w = None
    best_h = None
    
    # Enumerate all 2^N masks
    for mask_idx in range(2 ** N):
        # Convert mask index to binary vector
        h = np.array([(mask_idx >> i) & 1 for i in range(N)], dtype=float)
        
        # Optimize w for this mask
        w = optimize_w(np.random.randn(N), h, X, y, eta, K=K_adam, lr=1e-2)
        
        # Compute energy
        E = total_energy(w, h, X, y, eta, alpha, rho)
        
        if E < best_E:
            best_E = E
            best_w = w.copy()
            best_h = h.copy()
    
    return {
        'w': best_w,
        'h': best_h,
        'E': best_E,
        'h_binary': best_h.astype(int)
    }
