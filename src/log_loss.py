import numpy as np
from src.decorators import check_type_and_shape_vector_pair

@check_type_and_shape_vector_pair
def log_loss_(y, y_hat, eps=1e-15):
    """
    Computes the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    np.clip(y_hat, eps, 1 - eps, out=y_hat)
    m = y.shape[0]
    return (-(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))).item()