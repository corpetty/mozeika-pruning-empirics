"""
Multi-replica Glauber dynamics for Mozeika pruning.

The n = beta_h / beta_w parameter controls the effective temperature of the
h updates, with n < 1 promoting entropy maximization.
"""
import numpy as np


class MultiReplicaGlauber:
    """
    Multi-replica Glauber dynamics that maintains n independent weight chains
    {w_1, ..., w_n} sharing a common mask h.
    
    Each w_i optimizes with Adam independently given current h.
    The h update uses ensemble energy: E_ens = (1/n) * sum_i E(w_i, h|D)
    """
    
    def __init__(self, n_replicas, eta_val=0.0001, alpha=1.0, eta_h_reg=0.0001):
        """
        Initialize multi-replica Glauber.
        
        Args:
            n_replicas: Number of independent weight chains (n = beta_h/beta_w)
            eta_val: Weight regularization parameter
            alpha: Double-well potential parameter
            eta_h_reg: Mask regularization parameter
        """
        self.n = n_replicas
        self.eta = eta_val
        self.alpha = alpha
        self.eta_h = eta_h_reg
    
    def ensemble_energy(self, w_chains, h, X, y, eta_list, rho_list):
        """Compute ensemble energy across all replicas."""
        E_ens = 0
        for w in w_chains:
            E = self._single_energy(w, h, X, y, eta_list, rho_list)
            E_ens += E
        return E_ens / self.n
    
    def _single_energy(self, w, h, X, y, eta_list, rho_list):
        """Single replica energy."""
        n_layers = len(w)
        a = X
        for k in range(n_layers):
            z = a @ (w[k] * h[k])
            a = np.tanh(z) if k < n_layers - 1 else z
        L = np.mean((y - a.flatten()) ** 2) / 2
        reg = sum(eta * np.sum(wi ** 2) / 2 for wi, eta in zip(w, eta_list))
        V = sum(self.alpha * np.sum((h ** 2) * ((h - 1) ** 2)) + 
                rho * np.sum(h) / 2 for h, rho in zip(h, rho_list))
        return L + reg + V
    
    def run(self, w_init_chains, h_init, X, y, eta_list, rho_list, alpha_list, 
            T=50, T_h=1.0, rng=None):
        """
        Run multi-replica Glauber dynamics.
        
        Args:
            w_init_chains: List of n weight chains, each shape [(d0,d1), ..., (dL-1,dL)]
            h_init: Shared mask init, list of shape [h0, h1, ...]
            X, y: Training data
            eta_list: Per-layer eta regularization
            rho_list: Per-layer rho regularization
            alpha_list: Per-layer alpha (double-well)
            T: Number of Glauber sweeps
            T_h: Temperature for h updates (default 1.0 means standard Glauber)
            rng: Random number generator
        
        Returns:
            w_final: List of n pruned weight chains
            h_final: Shared pruned mask
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Deep copy initial weights
        w_chains = [[wi.copy() for wi in w_chain] for w_chain in w_init_chains]
        h = [hi.copy() for hi in h_init]
        
        n_layers = len(w_chains[0])
        losses = []
        
        for t in range(T):
            n_flips = 0
            
            # For each layer
            for l in range(n_layers):
                in_dim, out_dim = h[l].shape
                N = in_dim * out_dim
                
                # Coordinate updates with random order
                order = rng.permutation(N)
                
                for idx in order:
                    row = idx // out_dim
                    col = idx % out_dim
                    
                    # Collect all current weights for this layer
                    w_current = [w[l].copy() for w in w_chains]
                    
                    # Propose flipping bit j
                    h_try = h[l].copy()
                    h_try[row, col] = 1 - h_try[row, col]
                    h_try_list = h.copy()
                    h_try_list[l] = h_try
                    
                    # Each replica optimizes its weights for this proposed h
                    w_chains_try = []
                    for ci in range(self.n):
                        w_try = [wi.copy() for wi in w_chains[ci]]
                        w_try[l] = w_current[l].copy()  # Keep other layers same
                        
                        # Quick optimization for this mask
                        for _ in range(3):
                            w_try = self._adam_step(w_try, h_try_list, X, y, eta_list)
                        w_chains_try.append(w_try)
                    
                    # Compute ensemble energy diff
                    E_curr = self.ensemble_energy(w_chains, h, X, y, eta_list, rho_list)
                    E_try = self.ensemble_energy(w_chains_try, h_try_list, X, y, eta_list, rho_list)
                    
                    # Acceptance with temperature (n < 1 means T_h > 1)
                    delta_E = E_try - E_curr
                    if delta_E < 0:
                        accept = True
                    else:
                        accept = np.random.rand() < np.exp(-delta_E / T_h)
                    
                    if accept:
                        # Accept: update masks and weights
                        h[l] = h_try
                        for ci in range(self.n):
                            w_chains[ci][l] = w_chains_try[ci][l]
                        n_flips += 1
            
            # Global Adam re-optimization for all replicas
            for ci in range(self.n):
                w_chains[ci] = self._adam_chain(w_chains[ci], h, X, y, eta_list)
            
            # Track energy
            E = self.ensemble_energy(w_chains, h, X, y, eta_list, rho_list)
            losses.append(E)
        
        return w_chains, h, losses
    
    def _adam_step(self, w, h, X, y, eta_list):
        """Single-step Adam optimization for one replica."""
        n_layers = len(w)
        ms = [np.zeros_like(wi.flatten()) for wi in w]
        vs = [np.zeros_like(wi.flatten()) for wi in w]
        
        K = 5  # One-step Adam
        for k in range(K):
            grads = self._grad_chain(w, h, X, y)
            for l in range(n_layers):
                grads[l] = grads[l] + eta_list[l] * w[l]
            
            for l in range(n_layers):
                w_flat = w[l].flatten()
                g = grads[l].flatten()
                ms[l] = 0.9 * ms[l] + 0.1 * g
                vs[l] = 0.99 * vs[l] + 0.01 * g ** 2
                
                m_hat = ms[l] / (1 - 0.9 ** (k + 1))
                v_hat = vs[l] / (1 - 0.99 ** (k + 1))
                w_flat = w_flat - 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)
                w[l] = w_flat.reshape(w[l].shape)
        
        return w
    
    def _adam_chain(self, w, h, X, y, eta_list):
        """Run full Adam optimization for a weight chain."""
        n_layers = len(w)
        for _ in range(50):  # Multiple Adam steps per iteration
            w = self._adam_step(w, h, X, y, eta_list)
        return w
    
    def _grad_chain(self, w, h, X, y):
        """Backprop gradients for a weight chain."""
        M = X.shape[0]
        n_layers = len(w)
        
        # Forward
        a = X.copy()
        a_list = [X.copy()]
        z_list = []
        
        for k in range(n_layers):
            z = a @ (w[k] * h[k])
            z_list.append(z.copy())
            if k < n_layers - 1:
                a = np.tanh(z)
            else:
                a = z
            a_list.append(a.copy())
        
        # Backward
        grads = []
        delta = (a_list[-1] - y).flatten()
        
        for l in range(n_layers - 1, -1, -1):
            a_prev = a_list[l]
            delta_2d = delta.reshape(-1, 1)
            grad = a_prev.T @ delta_2d
            grad = grad * h[l]
            grads.insert(0, grad)
            
            if l > 0:
                delta_prev = delta_2d @ w[l].T
                delta_prev = delta_prev * (1 - z_list[l - 1] ** 2)
                delta = delta_prev.ravel()
        
        return grads


def multi_replica_glauber_sweep(N=100, n_list=[1, 2, 5, 10], eta=0.0001, rho=0.0007, 
                                  alpha=1.0, T=50, n_seeds=5):
    """
    Run multi-replica sweep at fixed (eta, rho).
    
    Args:
        N: Perceptron dimension
        n_list: List of replica counts to sweep
        eta: Weight regularization
        rho: Mask sparsity
        alpha: Double-well parameter
        T: Number of sweeps
        n_seeds: Random seeds for error bars
    
    Returns:
        results: List of Hamming at convergence per (n, seed)
    """
    results = []
    
    # Simple perceptron setup
    M = N  # One sample for simplicity
    X = np.random.randn(M, N).reshape(M, N)
    y = np.random.randn(M).reshape(M, 1)
    
    for n in n_list:
        print(f"  n={n}...")
        replica = MultiReplicaGlauber(n, eta_val=eta, alpha=alpha)
        
        h_s = []
        hamming_list = []
        
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)
            
            w_chains = []
            for _ in range(n):
                w_init = np.random.randn(N) * 0.1
                w_chains.append([w_init])
            
            h_init = [np.ones(N, dtype=float)]
            
            eta_list = [eta]
            rho_list = [rho]
            
            w_final, h_final, losses = replica.run(
                w_chains, h_init, X, y, eta_list, rho_list, alpha, T=50,
                T_h=1.0, rng=rng
            )
            
            # Compute Hamming (fraction of non-entries in mask)
            hamming = np.mean(h_final[0])
            hamming_list.append(hamming)
            h_s.append(hfinal := h_final[0].copy())
        
        # Aggregate
        mean_ham = np.mean(hamming_list)
        std_ham = np.std(hamming_list)
        results.append({
            'n': n,
            'mean_hamming': mean_ham,
            'std_hamming': std_ham,
            'seeds': hamming_list
        })
        print(f"    mean_hamming={mean_ham:.4f} +/- {std_ham:.4f}")
    
    return results
