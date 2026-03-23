import numpy as np


def relu(x):
    """ReLU activation."""
    return np.maximum(0, x)


def relu_grad(x):
    """ReLU derivative."""
    return (x > 0).astype(float)


def tanh_act(x):
    """Tanh activation."""
    return np.tanh(x)


def tanh_grad(x):
    """Tanh derivative."""
    return 1 - np.tanh(x) ** 2


def mlp_forward(w_list, h_list, X, activation='relu'):
    """
    Forward pass for MLP with masked weights.
    
    Architecture: input -> hidden1 -> hidden2 -> ... -> output
    
    Args:
        w_list: list of weight matrices, each (in_features, out_features)
        h_list: list of mask matrices, each (in_features, out_features)
        X: inputs (M, N_in)
        activation: 'relu', 'tanh', or 'identity'
    
    Returns:
        list of activations [a_0=X, a_1, a_2, ..., a_L=y_pred]
    """
    if activation == 'relu':
        phi = relu
        phi_grad = relu_grad
    elif activation == 'tanh':
        phi = tanh_act
        phi_grad = tanh_grad
    else:  # identity
        phi = lambda x: x
        phi_grad = lambda x: np.ones_like(x)
    
    activations = [X]  # a_0 = X
    
    a = X
    for w, h in zip(w_list, h_list):
        # Apply mask to weights
        w_masked = w * h
        # Linear transformation
        z = a @ w_masked  # (M, out_features)
        # Activation
        a = phi(z)
        activations.append(a)
    
    return activations


def mlp_loss(w_list, h_list, X, y, activation='relu'):
    """
    Squared loss for MLP.
    
    Args:
        w_list: list of weight matrices
        h_list: list of mask matrices
        X: inputs (M, N_in)
        y: targets (M,)
        activation: activation function for hidden layers
    
    Returns:
        scalar loss
    """
    activations = mlp_forward(w_list, h_list, X, activation)
    a_L = activations[-1]  # output
    return np.mean((y - a_L) ** 2) / 2


def double_well_mlp(h_list, alpha, rho_list):
    """
    Double-well potential summed over all layers.
    
    Args:
        h_list: list of mask matrices
        alpha: double-well barrier height
        rho_list: list of rho values per layer
    
    Returns:
        scalar potential energy
    """
    total = 0.0
    for h, rho in zip(h_list, rho_list):
        h_flat = h.flatten()
        total += np.sum(alpha * (h_flat ** 2) * ((h_flat - 1) ** 2) + (rho / 2) * h_flat)
    return total


def mlp_total_energy(w_list, h_list, X, y, eta_list, alpha, rho_list, activation='relu'):
    """
    Total MLP energy summed over layers.
    
    E = L + Σ_l [η_l||w_l||²/2 + Σ_i V(h_{l,i})]
    
    Args:
        w_list: list of weight matrices
        h_list: list of mask matrices
        X: inputs
        y: targets
        eta_list: list of eta values per layer
        alpha: double-well barrier height
        rho_list: list of rho values per layer
        activation: activation function
    
    Returns:
        scalar total energy
    """
    L = mlp_loss(w_list, h_list, X, y, activation)
    
    reg = 0.0
    for w, eta in zip(w_list, eta_list):
        reg += (eta / 2) * np.sum(w ** 2)
    
    V = double_well_mlp(h_list, alpha, rho_list)
    
    return L + reg + V


def mlp_grad_w(w_list, h_list, X, y, eta_list, activation='relu'):
    """
    Gradients ∂E/∂w_l for each layer via backprop.
    
    For layer l:
    ∂E/∂w_l = (X^(l-1)ᵀ @ δ_l) ∘ h_l + η_l * w_l
    
    Args:
        w_list: list of weight matrices
        h_list: list of mask matrices
        X: inputs (M, N_in)
        y: targets (M,)
        eta_list: list of eta values per layer
        activation: activation function
    
    Returns:
        list of gradient matrices (same shapes as w_list)
    """
    if activation == 'relu':
        phi = relu
        phi_grad = relu_grad
    elif activation == 'tanh':
        phi = tanh_act
        phi_grad = tanh_grad
    else:  # identity
        phi = lambda x: x
        phi_grad = lambda x: np.ones_like(x)
    
    M = X.shape[0]
    
    activations = [X]
    zs = []
    a = X
    for w, h in zip(w_list, h_list):
        w_masked = w * h
        z = a @ w
        zs.append(z)
        a = phi(z)
        activations.append(a)
    
    a_L = activations[-1]
    
    grad_w_list = []
    delta_l = (a_L - y) / M
    
    for l in range(len(w_list) - 1, -1, -1):
        grad_loss = activations[l].T @ delta_l
        grad_loss = grad_loss * h_list[l]
        grad_reg = eta_list[l] * w_list[l]
        grad_w = grad_loss + grad_reg
        grad_w_list.insert(0, grad_w)
        
        if l > 0:
            delta_l = (delta_l @ w_list[l].T) * phi_grad(zs[l - 1])
            delta_l = delta_l * h_list[l - 1]
            delta = delta_l
        else:
            delta = delta_l
    
    return grad_w_list


def grad_mlp_loss_w(w_list, h_list, X, y, activation='relu'):
    """
    Compute gradient of just the loss (no regularization).
    """
    if activation == 'relu':
        phi = relu
        phi_grad = relu_grad
    elif activation == 'tanh':
        phi = tanh_act
        phi_grad = tanh_grad
    else:
        phi = lambda x: x
        phi_grad = lambda x: np.ones_like(x)
    
    M = X.shape[0]
    
    activations = [X]
    zs = []
    a = X
    for w, h in zip(w_list, h_list):
        w_masked = w * h
        # handle 1D perceptron (N,) and 2D MLP (N_in, N_out) uniformly
        if w_masked.ndim == 1:
            z = a @ w_masked.reshape(-1, 1)
        else:
            z = a @ w_masked
        if z.ndim == 2 and z.shape[1] == 1:
            z = z.squeeze(1)
        zs.append(z)
        a = phi(z)
        activations.append(a)

    a_L = activations[-1]
    if a_L.ndim == 1:
        delta = (a_L - y.squeeze()) / M
    else:
        delta = (a_L - y) / M
    if delta.ndim == 1:
        delta = delta.reshape(-1, 1)

    grad_list = []

    for l in range(len(w_list) - 1, -1, -1):
        a_l = activations[l]
        if a_l.ndim == 1:
            a_l = a_l.reshape(-1, 1)
        grad_w = a_l.T @ delta          # shape matches w_list[l]
        grad_w = grad_w * h_list[l].reshape(grad_w.shape)
        # restore original shape if 1D
        grad_list.insert(0, grad_w.reshape(w_list[l].shape))

        if l > 0:
            w_l = w_list[l]
            if w_l.ndim == 1:
                w_l = w_l.reshape(-1, 1)
            z_prev = zs[l - 1]
            if z_prev.ndim == 1:
                z_prev = z_prev.reshape(-1, 1)
            delta = (delta @ w_l.T) * phi_grad(z_prev)
            delta = delta * h_list[l - 1].reshape(delta.shape)

    return grad_list


def mlp_sample(M, layer_sizes, sigma=0.01, seed=None):
    """
    Sample MLP data.
    
    Args:
        M: number of samples
        layer_sizes: list of layer sizes [N_in, N_h1, N_h2, ..., N_out]
        sigma: std for true weights
        seed: random seed
    
    Returns:
        X (M, N_in), y (M,), w0 (list of weight matrices), h0 (list of masks)
    """
    if seed is not None:
        np.random.seed(seed)
    
    N_in = layer_sizes[0]
    N_out = layer_sizes[-1]
    
    # Generate data
    X = np.random.randn(M, N_in)
    w0_list = []
    h0_list = []
    
    N_true = 0
    for i, (N_in_layer, N_out_layer) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        w = np.random.randn(N_in_layer, N_out_layer) * sigma
        h = (np.random.rand(N_in_layer, N_out_layer) > 0.5).astype(float)
        w0_list.append(w)
        h0_list.append(h)
        N_true += N_in_layer * N_out_layer
    
    # Compute targets
    a = X
    for w, h in zip(w0_list, h0_list):
        a = a @ w
        if i < len(w0_list) - 1:  # Not last layer
            a = np.tanh(a)
    y = a.flatten()  # Regression: output is already a vector
    
    # Add output mask (last layer is always on)
    h0_list[-1] = np.ones_like(h0_list[-1])
    
    return X, y, w0_list, h0_list


def mlp_glauber_step(w_list, h_list, X, y, eta_list, alpha, rho_list, activation='relu', rng=None):
    """
    Single Glauber sweep for one layer.
    
    For each neuron j in layer l:
        Flip h[l][j] if energy decreases
        Recompute gradients and optimize w after each flip
    
    Args:
        w_list: current weights
        h_list: current masks
        X: inputs
        y: targets
        eta_list: eta per layer
        alpha: double-well barrier
        rho_list: rho per layer
        activation: activation function
        rng: random number generator
    
    Returns:
        w_list, h_list (may be modified), flips (number of accepted flips)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    layer_sizes = [h.shape[0] for h in h_list] + [h_list[-1].shape[1]]
    
    total_flips = 0
    w_current = [w.copy() for w in w_list]
    h_current = [h.copy() for h in h_list]
    
    # Process layers sequentially
    for l in range(len(w_list)):
        in_dim, out_dim = h_current[l].shape
        N_l = in_dim * out_dim
        
        # Random order for this layer
        order = rng.permutation(N_l)
        
        for idx in order:
            j = idx // out_dim  # input neuron
            k = idx % out_dim   # output neuron
            
            # Try flipping h[l][j, k]
            h_try = h_current[l].copy()
            h_try[j, k] = 1 - h_try[j, k]
            
            # Try optimizing w for this modified mask
            w_try = [w.copy() for w in w_current]
            
            # Only re-optimize current layer's weights
            grad_loss = grad_mlp_loss_w(w_try, h_try, X, y, activation)
            lr = 1e-2
            K = 20
            for _ in range(K):
                grad_loss = grad_mlp_loss_w(w_try, h_try, X, y, activation)
                w_try[l] = w_try[l] - lr * grad_loss[l]
            
            # Compute energy difference
            E_current = mlp_total_energy(w_current, h_current, X, y, eta_list, alpha, rho_list, activation)
            E_try = mlp_total_energy(w_try, h_try, X, y, eta_list, alpha, rho_list, activation)
            
            if E_try < E_current:
                h_current[l] = h_try
                w_current[l] = w_try[l]
                total_flips += 1
    
    return w_current, h_current, total_flips
