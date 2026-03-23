import numpy as np


def sample_perceptron(N, M, p0, sigma=0.01, seed=None):
    """
    Generate synthetic data matching the R reference implementation.
    
    Data generation in R:
        D_N <- diag(1, N, N)
        zero <- rep(0, N)
        w0 <- mvrnorm(1, zero, D_N)  # w0 ~ N(0, I_N)
        N1 <- floor(N*p0)
        h0 <- sample(c(rep(1, N1), rep(0, N-N1)))  # exactly floor(N*p0) ones
        X <- mvrnorm(M, zero, D_N)/sqrt(N)  # X ~ N(0, I_N)/sqrt(N)
        noise <- mvrnorm(1, zero_M, D_M)
        y <- phi(X %*% (w0 * h0)) + sigma*noise  # phi=identity, so y = X @ (w0*h0) + noise
    
    Args:
        N: number of parameters
        M: number of training samples
        p0: target sparsity (fraction of ones in h0)
        sigma: noise standard deviation (default: 0.01)
        seed: random seed for reproducibility
    
    Returns:
        X: training inputs (M, N)
        y: training targets (M,)
        w0: true weights (N,)
        h0: true mask (N,)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Sample true weights: w0 ~ N(0, I_N)
    # R: mvrnorm(1, zero, D_N) where D_N = diag(1, N, N)
    w0 = rng.standard_normal(N)
    
    # Sample true mask: exactly floor(N*p0) ones, rest zeros
    # R: h0 <- sample(c(rep(1, N1), rep(0, N - N1)))
    N1 = int(np.floor(N * p0))
    h0_entries = np.concatenate([np.ones(N1), np.zeros(N - N1)])
    rng.shuffle(h0_entries)
    h0 = h0_entries
    
    # Sample training inputs: X ~ N(0, I_N)/sqrt(N)
    # R: mvrnorm(M, zero, D_N)/sqrt(N)
    X = rng.standard_normal((M, N)) / np.sqrt(N)
    
    # Sample noise
    noise = rng.standard_normal(M)
    
    # Compute targets: y = X @ (w0 * h0) + sigma * noise
    # Since phi is identity
    y = X @ (w0 * h0) + sigma * noise
    
    return X, y, w0, h0


def sample_perceptron_test(N, M_test, seed=None):
    """
    Generate test data matching the R reference implementation.
    
    Args:
        N: number of parameters
        M_test: number of test samples
        seed: random seed
    
    Returns:
        X_test: test inputs (M_test, N)
        y_test: test targets (M_test,)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Sample training inputs: X_test ~ N(0, I_N)/sqrt(N)
    X_test = rng.standard_normal((M_test, N)) / np.sqrt(N)
    
    return X_test


def sample_data_batch(N, M, M_test, p0, sigma, eta, rho, alpha, seed):
    """
    Generate full dataset for one experiment batch (matching R's single run).
    
    Args:
        N: number of parameters
        M: training samples
        M_test: test samples
        p0: target sparsity
        sigma: noise level
        eta: L2 regularization
        rho: sparsity pressure
        alpha: double-well barrier
        seed: random seed
    
    Returns:
        X, y: training data
        X_test, y_test: test data
        w0, h0: true parameters
    """
    # Sample true parameters
    rng_train = np.random.default_rng(seed)
    w0 = rng_train.standard_normal(N)
    N1 = int(np.floor(N * p0))
    h0_entries = np.concatenate([np.ones(N1), np.zeros(N - N1)])
    rng_train.shuffle(h0_entries)
    h0 = h0_entries
    
    # Sample training data
    X = rng_train.standard_normal((M, N)) / np.sqrt(N)
    noise = rng_train.standard_normal(M)
    y = X @ (w0 * h0) + sigma * noise
    
    # Sample test data (deterministic based on true params)
    rng_test = np.random.default_rng(seed + 1)  # Different seed for test
    X_test = rng_test.standard_normal((M_test, N)) / np.sqrt(N)
    y_test = X_test @ (w0 * h0) + sigma * rng_test.standard_normal(M_test)
    
    return X, y, X_test, y_test, w0, h0
