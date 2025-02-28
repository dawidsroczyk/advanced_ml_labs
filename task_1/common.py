import numpy as np


def empirical_covariance(X: np.array) -> np.array:
    X_diff_mean = (X - X.mean(axis=0)).T
    S = np.linalg.matmul(X_diff_mean, X_diff_mean.T)
    S = S / (X.shape[0] - 1)
    return S


def naive_covariance(X: np.array) -> np.array:
    result = np.sum((X - X.mean(axis=0))**2, axis=0) / (X.shape[0] - 1)
    result = np.diag(result)
    return result