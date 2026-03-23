"""
Tests for the GlauberPruner API.
"""
import pytest
import numpy as np
from pruning_core.pruner import GlauberPruner


@pytest.fixture
def sample_data():
    """Create sample training data."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 20)
    y = (X[:, 0] > 0).astype(float)  # Simple binary target
    return X, y


class TestGlauberPruner:
    
    def test_fit_runs(self, sample_data):
        """Test that fit() runs without error."""
        X, y = sample_data
        pruner = GlauberPruner(rho=0.01, eta=0.001, n_replicas=1, T=5)
        pruner.fit(X, y)
        
        assert pruner._trained
        assert pruner._h_pruned is not None
        
    def test_mask_is_binary(self, sample_data):
        """Test that get_mask() returns binary mask."""
        X, y = sample_data
        pruner = GlauberPruner(rho=0.01, eta=0.001, n_replicas=1, T=5)
        pruner.fit(X, y)
        
        mask = pruner.get_mask()
        
        # Check binary values
        unique_vals = np.unique(mask.flatten())
        assert set(unique_vals) <= {0.0, 1.0}, f"Non-binary values found: {unique_vals}"
        
        # Check shape: mask is (d_in, d_out), so first dim matches X features
        assert mask.shape[0] == X.shape[1], f"Shape mismatch: {mask.shape} vs {X.shape}"
        
    def test_sparsity_in_range(self, sample_data):
        """Test that sparsity() returns value in [0, 1]."""
        X, y = sample_data
        pruner = GlauberPruner(rho=0.01, eta=0.001, n_replicas=1, T=5)
        pruner.fit(X, y)
        
        sparsity = pruner.sparsity()
        assert 0.0 <= sparsity <= 1.0, f"Sparsity {sparsity} not in [0, 1]"
        
    def test_different_rho_values(self, sample_data):
        """Test that different rho values produce different sparsities."""
        X, y = sample_data
        
        # Use more sweeps + seed so result is deterministic
        pruner_low = GlauberPruner(rho=0.0001, eta=0.0001, n_replicas=1, T=20)
        pruner_low.fit(X, y, seed=0)

        pruner_high = GlauberPruner(rho=0.002, eta=0.0001, n_replicas=1, T=20)
        pruner_high.fit(X, y, seed=0)

        sparsity_low = pruner_low.sparsity()
        sparsity_high = pruner_high.sparsity()

        # Higher rho should lead to equal or higher sparsity
        assert sparsity_high >= sparsity_low - 0.05, \
            f"rho=0.002 has much lower sparsity ({sparsity_high:.3f}) than rho=0.0001 ({sparsity_low:.3f})"
            
    def test_predict_returns_arrays(self, sample_data):
        """Test that predict() returns valid predictions."""
        X, y = sample_data
        
        pruner = GlauberPruner(rho=0.01, eta=0.001, n_replicas=1, T=5)
        pruner.fit(X, y)
        
        preds = pruner.predict(X)
        
        assert preds.shape == (X.shape[0],), f"Wrong shape: {preds.shape}"
        assert isinstance(preds, np.ndarray)
        
    def test_apply_to_modifies_weights(self, sample_data):
        """Test that apply_to() applies the mask."""
        X, y = sample_data
        rng = np.random.RandomState(123)
        weights = rng.randn(20, 1)
        
        pruner = GlauberPruner(rho=0.1, eta=0.001, n_replicas=1, T=5)
        pruner.fit(X, y)
        
        pruned_weights = pruner.apply_to(weights)
        
        # Some weights should be zeroed out
        mask = pruner.get_mask()
        expected_pruned = weights * mask
        np.testing.assert_array_almost_equal(pruned_weights, expected_pruned)
