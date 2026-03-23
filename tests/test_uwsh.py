"""
Tests for UWSH subspace analysis (Experiment 19).

Tests the spectral analysis helpers and verifies that pruning
concentrates weights into shared directions.
"""
import pytest
import numpy as np
import sys

sys.path.insert(0, '/home/petty/pruning-research')

from experiments.uwsh_subspace_helpers import (
    participation_ratio,
    top_k_variance_fraction,
    mean_pairwise_cosine_similarity,
    spectral_analysis,
    run_glauber_single,
)
from pruning_core.data import sample_perceptron


class TestParticipationRatio:

    def test_uniform_eigenvalues(self):
        """Uniform eigenvalues → PR = N (maximally spread)."""
        eigvals = np.ones(20)
        pr = participation_ratio(eigvals)
        assert abs(pr - 20.0) < 0.01

    def test_single_dominant(self):
        """One dominant eigenvalue → PR ≈ 1."""
        eigvals = np.array([100.0] + [0.001] * 19)
        pr = participation_ratio(eigvals)
        assert pr < 1.5

    def test_random_vectors_pr_near_half_n(self):
        """PR of random weight vectors should be near N/2 (not N, because
        we have fewer runs than dimensions)."""
        rng = np.random.default_rng(42)
        N = 60
        n_runs = 20
        W = rng.normal(0, 1, (n_runs, N))
        gram = W @ W.T
        eigvals = np.linalg.eigvalsh(gram)[::-1]
        pr = participation_ratio(eigvals)
        # For 20 random vectors in R^60, PR should be near n_runs
        # (all n_runs eigenvalues of the gram matrix are comparable)
        assert pr > n_runs * 0.5, f"PR={pr:.1f} too low for random vectors"


class TestTopKVariance:

    def test_all_variance_in_top1(self):
        """If S = [10, 0, 0, ...], top-1 variance = 1.0."""
        S = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
        frac = top_k_variance_fraction(S, k=1)
        assert abs(frac - 1.0) < 0.01

    def test_uniform_singular_values(self):
        """If all singular values equal, top-5 out of 20 → 0.25."""
        S = np.ones(20)
        frac = top_k_variance_fraction(S, k=5)
        assert abs(frac - 0.25) < 0.01


class TestCosineSimilarity:

    def test_identical_vectors(self):
        """Identical rows → cos_sim = 1.0."""
        W = np.tile([1.0, 2.0, 3.0], (5, 1))
        cs = mean_pairwise_cosine_similarity(W)
        assert abs(cs - 1.0) < 0.01

    def test_orthogonal_vectors(self):
        """Orthogonal rows → cos_sim ≈ 0."""
        W = np.eye(10)[:5]  # 5 orthogonal unit vectors in R^10
        cs = mean_pairwise_cosine_similarity(W)
        assert cs < 0.05


class TestPruningConcentration:
    """Integration tests: verify that pruning concentrates weights."""

    @pytest.fixture(scope='class')
    def pruned_weights(self):
        """Run Glauber at rho=0 and rho=high, collect pruned weights."""
        N = 40
        M = 120
        n_seeds = 8
        X, y, w0, h0 = sample_perceptron(N, M, p0=0.5, sigma=0.01, seed=0)

        results = {}
        for rho in [0.0, 0.0005]:
            W = []
            for seed in range(n_seeds):
                w_final, h_final = run_glauber_single(
                    X, y, eta=1e-4, rho=rho, alpha=1.0, T=15,
                    seed=seed + 2000
                )
                W.append(w_final * h_final)
            results[rho] = np.array(W)
        return results

    def test_pr_decreases_with_rho(self, pruned_weights):
        """Participation ratio should decrease as rho increases."""
        metrics_0 = spectral_analysis(pruned_weights[0.0])
        metrics_high = spectral_analysis(pruned_weights[0.0005])
        assert metrics_high['participation_ratio'] < metrics_0['participation_ratio'], \
            (f"PR at rho=0.0005 ({metrics_high['participation_ratio']:.1f}) "
             f"should be < PR at rho=0 ({metrics_0['participation_ratio']:.1f})")

    def test_cos_sim_increases_with_rho(self, pruned_weights):
        """Cosine similarity should increase as rho increases."""
        metrics_0 = spectral_analysis(pruned_weights[0.0])
        metrics_high = spectral_analysis(pruned_weights[0.0005])
        assert metrics_high['mean_pairwise_cos_sim'] > metrics_0['mean_pairwise_cos_sim'], \
            (f"cos_sim at rho=0.0005 ({metrics_high['mean_pairwise_cos_sim']:.4f}) "
             f"should be > cos_sim at rho=0 ({metrics_0['mean_pairwise_cos_sim']:.4f})")
