"""Tests for MLP energy module."""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pruning_core.energy_mlp import (
    mlp_forward, mlp_loss, mlp_total_energy,
    mlp_glauber_step, relu, mlp_sample
)


def make_mlp(layer_sizes, M=50, seed=42):
    rng = np.random.default_rng(seed)
    w_list = [rng.standard_normal((layer_sizes[i], layer_sizes[i+1])) * 0.3
              for i in range(len(layer_sizes)-1)]
    h_list = [np.ones((layer_sizes[i], layer_sizes[i+1]))
              for i in range(len(layer_sizes)-1)]
    X = rng.standard_normal((M, layer_sizes[0]))
    return w_list, h_list, X


def test_mlp_forward_shape():
    """Forward pass returns activations for each layer with correct final shape."""
    layer_sizes = [4, 8, 4, 2]
    M = 30
    w_list, h_list, X = make_mlp(layer_sizes, M)
    out = mlp_forward(w_list, h_list, X)
    # mlp_forward returns list of activations; last one is final output
    if isinstance(out, list):
        final = out[-1]
    else:
        final = out
    assert final.shape == (M, layer_sizes[-1]), f"Expected ({M}, {layer_sizes[-1]}), got {final.shape}"


def test_mlp_energy_nonneg():
    """Total energy is non-negative."""
    layer_sizes = [4, 8, 1]
    M = 50
    w_list, h_list, X = make_mlp(layer_sizes, M)
    rng = np.random.default_rng(0)
    y = rng.standard_normal((M, 1))
    eta_list = [0.001] * (len(layer_sizes) - 1)
    rho_list = [0.0005] * (len(layer_sizes) - 1)
    E = mlp_total_energy(w_list, h_list, X, y, eta_list, alpha=1.0, rho_list=rho_list)
    assert E >= 0, f"Energy should be >= 0, got {E}"


def test_mlp_forward_zero_mask():
    """With all-zero mask, output should be zero (dead network)."""
    layer_sizes = [4, 8, 1]
    M = 20
    w_list, h_list, X = make_mlp(layer_sizes, M)
    h_zero = [np.zeros_like(h) for h in h_list]
    out = mlp_forward(w_list, h_zero, X)
    # mlp_forward returns list of activations or final array
    final = out[-1] if isinstance(out, list) else out
    assert np.allclose(final, 0), "Zero mask should give zero output"


def test_mlp_layer_count():
    """Number of weight matrices matches number of layer transitions."""
    layer_sizes = [4, 8, 4, 1]
    w_list, h_list, X = make_mlp(layer_sizes)
    assert len(w_list) == len(layer_sizes) - 1
    assert len(h_list) == len(layer_sizes) - 1
    for i, (w, h) in enumerate(zip(w_list, h_list)):
        assert w.shape == (layer_sizes[i], layer_sizes[i+1])
        assert h.shape == (layer_sizes[i], layer_sizes[i+1])
