"""
Tests for finite-temperature Glauber dynamics.
"""
import numpy as np
import sys
sys.path.insert(0, '/home/petty/pruning-research')

from pruning_core.data import sample_perceptron
from pruning_core.dynamics import run_glauber, run_glauber_finite_temp
from pruning_core.metrics import hamming_distance


def test_finite_temp_more_flips_than_zero_temp():
    """
    At the same rho, finite-temperature Glauber should accept at least as
    many flips as zero-temperature (greedy), because it accepts uphill moves
    with probability exp(-ΔE/T_h) > 0.
    """
    N, M = 15, 45
    X, y, w0, h0 = sample_perceptron(N, M, p0=0.5, sigma=0.01, seed=42)

    w_init = np.random.default_rng(0).normal(0, 0.1, N)
    h_init = np.ones(N, dtype=float)
    rho = 0.0005  # moderate rho — some flips accepted at zero-T
    alpha = 1.0

    # Zero-temperature (greedy)
    rng_cold = np.random.default_rng(99)
    res_cold = run_glauber(w_init, h_init, X, y, eta=1e-4, rho=rho,
                           alpha=alpha, T=5, rng=rng_cold)
    flips_cold = sum(res_cold['history']['flips'])

    # Finite temperature (warm) — high T_h to ensure exploration
    rng_warm = np.random.default_rng(99)
    res_warm = run_glauber_finite_temp(w_init, h_init, X, y, eta=1e-4,
                                        rho=rho, alpha=alpha, T=5,
                                        T_h=0.5, rng=rng_warm)
    flips_warm = sum(res_warm['history']['flips'])

    # Warm should accept at least as many flips as cold
    assert flips_warm >= flips_cold, (
        f"Expected warm ({flips_warm}) >= cold ({flips_cold})"
    )


def test_finite_temp_low_T_recovers_zero_temp():
    """
    At very low T_h (→0), finite-temperature Glauber should produce
    the same mask as zero-temperature (greedy) Glauber, because
    exp(-ΔE/T_h) → 0 for any ΔE > 0.
    """
    N, M = 15, 45
    X, y, w0, h0 = sample_perceptron(N, M, p0=0.5, sigma=0.01, seed=7)

    w_init = np.random.default_rng(1).normal(0, 0.1, N)
    h_init = np.ones(N, dtype=float)
    rho = 0.0005
    alpha = 1.0

    # Zero-temperature
    res_cold = run_glauber(w_init, h_init, X, y, eta=1e-4, rho=rho,
                           alpha=alpha, T=8, rng=np.random.default_rng(50))
    hd_cold = hamming_distance(res_cold['h'], h0)

    # Very low temperature (effectively greedy)
    res_ft = run_glauber_finite_temp(w_init, h_init, X, y, eta=1e-4,
                                      rho=rho, alpha=alpha, T=8,
                                      T_h=1e-12, rng=np.random.default_rng(50))
    hd_ft = hamming_distance(res_ft['h'], h0)

    # Should produce similar Hamming distance (not identical due to RNG
    # state divergence from extra random() draws, but within tolerance)
    assert abs(hd_cold - hd_ft) < 0.15, (
        f"Expected similar Hamming: cold={hd_cold:.3f}, finite_temp={hd_ft:.3f}"
    )


def test_multi_replica_finite_temp_basic():
    """
    Basic smoke test for multi_replica_glauber_finite_temp:
    should return a Hamming distance in [0, 1].
    """
    from pruning_core.replicas import multi_replica_glauber_finite_temp

    N, M = 10, 30
    X, y, w0, h0 = sample_perceptron(N, M, p0=0.5, sigma=0.01, seed=3)

    hd = multi_replica_glauber_finite_temp(
        X, y, h0, eta=1e-4, rho=0.0005, alpha=1.0,
        n_replicas=2, T=3, T_h=0.01, seed=42
    )

    assert 0.0 <= hd <= 1.0, f"Hamming distance out of range: {hd}"


if __name__ == '__main__':
    print("test_finite_temp_more_flips_than_zero_temp...", end=' ', flush=True)
    test_finite_temp_more_flips_than_zero_temp()
    print("PASS")

    print("test_finite_temp_low_T_recovers_zero_temp...", end=' ', flush=True)
    test_finite_temp_low_T_recovers_zero_temp()
    print("PASS")

    print("test_multi_replica_finite_temp_basic...", end=' ', flush=True)
    test_multi_replica_finite_temp_basic()
    print("PASS")

    print("\nAll finite-temp tests passed!")
