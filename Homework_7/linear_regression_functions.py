import numpy as np


def sum_of_squares(samples):
    return ((samples - samples.mean()) ** 2).sum()


def standard_error_slope(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
) -> float:
    n = x.shape[0]
    upper = ((y - z) ** 2).sum() / (n - 2)
    lower = ((x - x.mean()) ** 2).sum()
    return np.sqrt(upper / lower)


def standard_error_intercept(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
) -> float:
    return standard_error_slope(x, y, z) * np.sqrt((x ** 2).mean())

