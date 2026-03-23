import numpy as np


def hamming_distance(h_est, h_true):
    """
    Compute normalized Hamming distance (fraction of misclassified bits).
    
    Matches R's: sum((h1 - h0)^2) / N
    
    Args:
        h_est: estimated mask
        h_true: true mask
    
    Returns:
        normalized Hamming distance
    """
    return np.sum((h_est - h_true) ** 2) / len(h_true)


def mse_w(w_est, w_true):
    """
    Compute mean squared error for weights.
    
    Matches R's: sum((w1*h1 - w0*h0)^2) / N
    
    Args:
        w_est: estimated weights (with mask applied)
        w_true: true weights (with mask applied)
    
    Returns:
        MSE
    """
    return np.sum((w_est - w_true) ** 2) / len(w_true)


def sparsity(h):
    """
    Compute sparsity fraction (fraction of zeros).
    
    Args:
        h: mask (binary)
    
    Returns:
        sparsity (fraction of zeros)
    """
    return np.sum(h == 0) / len(h)


def sparsity_ratio(h):
    """
    Compute the opposite of sparsity: fraction of ones (non-zero).
    
    Args:
        h: mask
    
    Returns:
        fraction of ones
    """
    return np.mean(h)
