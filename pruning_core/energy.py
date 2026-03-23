import numpy as np


def squared_loss(w, h, X, y, phi=None):
    """
    L(w ∘ h | D) = (1/2M) * sum((y - phi(X @ (w ∘ h)))^2)
    
    Args:
        w: weights (N,)
        h: binary mask (N,)
        X: inputs (M, N)
        y: targets (M,)
        phi: activation function (default: identity)
    
    Returns:
        scalar loss
    """
    if phi is None:
        phi = lambda x: x
    
    w_h = w * h
    preds = phi(X @ w_h)
    return np.mean((y - preds) ** 2) / 2


def double_well(h, alpha, rho=None):
    """
    V(h) = alpha * h^2 * (h - 1)^2 + (rho/2) * h
    
    This is a double-well potential with minima at h=0 and h=1.
    The rho term adds a bias toward h=0 (sparsity).
    
    Args:
        h: binary or continuous mask
        alpha: double-well barrier height
        rho: sparsity pressure
    
    Returns:
        V(h) summed over all elements
    """
    if rho is None:
        rho = 0.0
    return np.sum(alpha * (h ** 2) * ((h - 1) ** 2) + (rho / 2) * h)


def total_energy(w, h, X, y, eta, alpha, rho, phi=None):
    """
    E(w, h | D) = L(w ∘ h | D) + (eta/2) * ||w||^2 + sum(V(h_i))
    
    Args:
        w: weights
        h: binary mask
        X: inputs
        y: targets
        eta: L2 regularization coefficient
        alpha: double-well barrier
        rho: sparsity pressure
        phi: activation function
    """
    if phi is None:
        phi = lambda x: x
    
    L_wh = squared_loss(w, h, X, y, phi)
    reg = (eta / 2) * np.sum(w ** 2)
    V_h = double_well(h, alpha, rho)
    
    return L_wh + reg + V_h


def grad_energy_w(w, h, X, y, eta, phi=None):
    """
    Compute gradient ∂E/∂w for the energy function.
    
    For φ(x) = x (identity):
        ∂L/∂w = Xᵀ(X @ w - y) ∘ h
        ∂(η/2||w||²)/∂w = η * w
    
    Args:
        w: weights
        h: binary mask
        X: inputs (M, N)
        y: targets (M,)
        eta: L2 regularization coefficient
        phi: activation function (default: identity)
    
    Returns:
        gradient vector (N,)
    """
    if phi is None:
        phi = lambda x: x
    
    # w_h = w * h (pruned weights)
    w_h = w * h
    
    # Forward pass
    preds = phi(X @ w_h)
    residuals = preds - y
    
    # Backward pass for loss gradient
    # For identity: dL/d(w*h) = Xᵀ(residuals) / M
    # Then chain rule: dL/dw = dL/d(w*h) * h
    grad_loss = (X.T @ residuals) / len(y)
    
    # Multiply by h (chain rule for w*h)
    grad_loss = grad_loss * h
    
    # Regularization gradient
    grad_reg = eta * w
    
    return grad_loss + grad_reg


def grad_energy_w_tanh(w, h, X, y, eta):
    """
    Compute gradient ∂E/∂w for φ(x) = tanh(x).
    
    dL/dw = [Xᵀ((w*h) * (1 - (X@(w*h))²)) * residuals] * h + η*w
    
    Args:
        w: weights
        h: binary mask
        X: inputs (M, N)
        y: targets (M,)
        eta: L2 regularization coefficient
    
    Returns:
        gradient vector (N,)
    """
    w_h = w * h
    preds = np.tanh(X @ w_h)
    residuals = preds - y
    
    # d/d(w*h) of loss with tanh
    grad_pre = (1 - preds ** 2) * residuals / len(y)
    
    # Apply chain rule for w*h
    grad_w = (X.T @ grad_pre) * h
    
    # Add regularization
    grad_w = grad_w + eta * w
    
    return grad_w
