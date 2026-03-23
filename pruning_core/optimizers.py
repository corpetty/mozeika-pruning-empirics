import numpy as np


class AdamOptimizer:
    """
    Adam optimizer matching the R implementation exactly.
    
    R implementation:
        state$t <- state$t + 1L
        state$m <- beta1 * state$m + (1 - beta1) * gw
        state$v <- beta2 * state$v + (1 - beta2) * (gw * gw)
        m_hat <- state$m / (1 - beta1^state$t)
        v_hat <- state$v / (1 - beta2^state$t)
        w_new  <- w - lr * m_hat / (sqrt(v_hat) + eps)
    
    Default hyperparameters match R code:
        lr=1e-2, β1=0.9, β2=0.999, ε=1e-8
    """
    
    def __init__(self, N, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Initialize Adam optimizer.
        
        Args:
            N: number of parameters
            lr: learning rate (default: 1e-2 to match R)
            beta1: exponential decay rate for first moment (default: 0.9)
            beta2: exponential decay rate for second moment (default: 0.999)
            eps: small constant for numerical stability (default: 1e-8)
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # Initialize moments and timestep
        self.m = np.zeros(N)
        self.v = np.zeros(N)
        self.t = 0
    
    def step(self, w, grad):
        """
        Perform one Adam optimization step.
        
        Args:
            w: current weights
            grad: gradient
        
        Returns:
            new weights
        """
        # Increment timestep
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update weights
        w_new = w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        return w_new


def optimize_w(w_init, h, X, y, eta, K=50, lr=1e-2, phi=None):
    """
    Run K Adam optimization steps to optimize w given fixed h.
    
    Matches R's optimize_w_adam function.
    
    Args:
        w_init: initial weights
        h: binary mask (fixed during optimization)
        X: inputs (M, N)
        y: targets (M,)
        eta: L2 regularization coefficient
        K: number of optimization steps (default: 50)
        lr: learning rate (default: 1e-2)
        phi: activation function (default: identity)
    
    Returns:
        optimized w
    """
    if phi is None:
        phi = lambda x: x
    
    w = w_init.copy()
    
    # Initialize optimizer for this problem
    N = len(w)
    adam = AdamOptimizer(N, lr=lr, beta1=0.9, beta2=0.999, eps=1e-8)
    
    for k in range(K):
        # Compute gradient
        grad = grad_energy_w(w, h, X, y, eta)
        
        # Adam step
        w = adam.step(w, grad)
    
    return w


def grad_energy_w_fn(w, h, X, y, eta, phi=None):
    """
    Get gradient function for given (w, h, X, y, eta).
    This is a helper for use with optimizers.
    """
    def grad_fn(w):
        return grad_energy_w(w, h, X, y, eta)
    return grad_fn


# Re-export for module compatibility
from .energy import grad_energy_w
