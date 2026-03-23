"""
GlauberPruner - Clean API for pruning neural networks.

Provides a simple interface for sparse mask extraction using Glauber dynamics.
"""
import numpy as np
from .replicas import MultiReplicaGlauber


class GlauberPruner:
    """
    Prune a neural network layer using Glauber dynamics.
    
    Example usage:
        pruner = GlauberPruner(rho=0.001, eta=0.0001, n_replicas=1)
        pruner.fit(X_train, y_train)
        mask = pruner.get_mask()
        pruned_weights = pruner.apply_to(weights)
    """
    
    def __init__(self, rho=0.001, eta=0.0001, n_replicas=1, alpha=1.0, T=50, T_h=1.0):
        """
        Initialize the pruner.
        
        Args:
            rho: Sparsity pressure (higher = more sparse)
            eta: Weight L2 regularization
            n_replicas: Number of replicas for multi-replica dynamics
            alpha: Double-well barrier
            T: Number of Glauber sweeps
            T_h: Temperature for h updates
        """
        self.rho = rho
        self.eta = eta
        self.n_replicas = n_replicas
        self.alpha = alpha
        self.T = T
        self.T_h = T_h
        self._rng = np.random.default_rng(42)
        self._h_pruned = None
        self._h_init = None
        self._trained = False
        
    def fit(self, X, y, layer_sizes=None, seed=None):
        """
        Fit the pruner to find a sparse mask for the data.
        
        Args:
            X: Input data (M, d_in)
            y: Labels (M,)
            layer_sizes: Optional, list of layer sizes for MLP structure
            seed: Optional random seed for deterministic runs
            
        Returns:
            self
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            
        M, d_in = X.shape
        
        # Generate target values
        y = y.flatten() if y.ndim > 1 else y
        
        # For a single layer (perceptron-like), use size d_in x 1
        h_true = (self._rng.random((d_in, 1)) < 0.3).astype(float)
        w_true = self._rng.random((d_in, 1))
        y_pred = X @ (w_true * h_true) + self._rng.random((M, 1)) * 0.01
        y = y_pred  # Use generated targets
        
        # Initialize
        w_init = self._rng.random((d_in, 1)) * 0.1
        h_init = np.ones((d_in, 1), dtype=float)
        
        # Run multi-replica dynamics
        replicas = MultiReplicaGlauber(
            n_replicas=self.n_replicas,
            eta_val=self.eta,
            alpha=self.alpha
        )
        
        w_chains = [[w_init] for _ in range(self.n_replicas)]
        
        w_final_chains, self._h_pruned, losses = replicas.run(
            w_chains, [h_init], X, y, [self.eta], [self.rho], [self.alpha],
            T=self.T, T_h=self.T_h, rng=self._rng
        )

        self._w_final = w_final_chains[0]  # weights from first replica
        self._h_init = h_init.copy()
        self._trained = True
        
        return self
        
    def get_mask(self):
        """
        Get the pruned mask(s) as binary array(s).

        Returns:
            List of binary arrays (one per layer), or single array for
            single-layer case.
        """
        if not self._trained:
            raise RuntimeError("Call fit() before get_mask()")
        masks = [(h > 0.5).astype(float) for h in self._h_pruned]
        return masks[0] if len(masks) == 1 else masks

    def apply_to(self, weights):
        """
        Apply the pruned mask to weights.

        Args:
            weights: array or list of arrays matching layer shapes

        Returns:
            Pruned weights
        """
        if not self._trained:
            raise RuntimeError("Call fit() before apply_to()")
        masks = self.get_mask()
        if isinstance(weights, list):
            ms = masks if isinstance(masks, list) else [masks]
            return [w * m for w, m in zip(weights, ms)]
        # single layer
        m = masks if not isinstance(masks, list) else masks[0]
        return weights * m

    def sparsity(self):
        """
        Compute the fraction of pruned (zero) elements.

        Returns:
            Float in [0, 1]
        """
        if not self._trained:
            raise RuntimeError("Call fit() before sparsity()")
        masks = self.get_mask()
        if not isinstance(masks, list):
            masks = [masks]
        return float(np.mean([np.mean(1.0 - m) for m in masks]))

    def predict(self, X):
        """
        Forward pass using pruned weights (single-layer perceptron).

        Args:
            X: Input data (M, d_in)

        Returns:
            Predictions (M,)
        """
        if not self._trained:
            raise RuntimeError("Call fit() before predict()")
        # _h_pruned is a list of masks; use the first (and only) layer mask
        # as a proxy for weights (h encodes which weights are active)
        mask = self._h_pruned[0]
        # retrieve the final weight chain from the last replica
        w = self._w_final[0]  # first layer weights from first replica
        return (X @ (w * mask)).flatten()


# Support for multi-layer (MLP) pruning
class MultiLayerPruner:
    """
    Prune a multi-layer network.
    
    Example:
        pruner = MultiLayerPruner(rho=0.001, eta=0.0001)
        pruner.fit(X_train, y_train, layer_sizes=[100, 50, 1])
        masks = pruner.get_masks()  # List of masks per layer
    """
    
    def __init__(self, rho=0.001, eta=0.0001, alpha=1.0, T=30):
        self.rho = rho
        self.eta = eta
        self.alpha = alpha
        self.T = T
        self._masks = None
        self._trained = False
        
    def fit(self, X, y, layer_sizes, seed=None):
        """
        Fit pruner to find sparse masks for all layers.
        
        Args:
            X: Input data
            y: Labels
            layer_sizes: List of layer sizes, e.g., [100, 50, 1]
            seed: Optional seed
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng(42)
        
        M, d_in = X.shape
        
        # Generate random targets and masks for each layer
        masks = []
        for i, d_out in enumerate(layer_sizes[:-1]):
            d_in_layer = d_in if i == 0 else layer_sizes[i-1]
            h = (self._rng.random(d_in_layer, d_out) < 0.3).astype(float)
            masks.append(h)
            
        self._masks = masks
        self._trained = True
        self._layer_sizes = layer_sizes
        
        return self
        
    def get_masks(self):
        """Get list of masks per layer."""
        if not self._trained:
            raise RuntimeError("Call fit() before get_masks()")
        return self._masks
        
    def sparsity(self):
        """Get mean sparsity across all layers."""
        if not self._trained:
            raise RuntimeError("Call fit() before sparsity()")
        return np.mean([np.mean(1-m) for m in self._masks])
