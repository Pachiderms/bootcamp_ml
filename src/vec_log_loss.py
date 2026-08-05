import numpy as np
from src.decorators import check_type_and_shape_vector_pair


@check_type_and_shape_vector_pair
def vec_log_loss_(y, y_hat, eps=1e-15):
    """
    Computes the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: epsilon (default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    m = y.shape[0]
    np.clip(y_hat, eps, 1 - eps, out=y_hat)
    sum = np.dot(y.T, np.log(y_hat)) + np.dot(
        (np.ones((m, 1)) - y).T, np.log(np.ones((m, 1)) - y_hat)
    )
    return -(1 / m) * sum.item()
