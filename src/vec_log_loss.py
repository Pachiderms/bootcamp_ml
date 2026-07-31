import numpy as np

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
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        print(f"type err: {type(y)=} {type(y_hat)=}")
        return None
    m = y.shape[0]
    if (y.shape, y_hat.shape) != ((m, 1), (m, 1)):
        print(f"shape err: {y.shape=} {y_hat.shape=}")
        return None

    np.clip(y_hat, eps, 1 - eps)
    return -(1 / m) * (np.dot(y.T, np.log(y_hat)) + np.dot((np.ones((m, 1)) - y).T, np.log(np.ones((m, 1)) - y_hat)))