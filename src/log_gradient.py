import numpy as np
from src.log_pred import logistic_predict_

def log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatibl
    Args:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
            print(f"type err: {type(x)=} {type(y)=} {type(theta)=}")
            return None

    m, n = x.shape
    if y.shape != (m, 1) or theta.shape != (n + 1, 1):
        print(f"shape err: {x.shape=} {y.shape=} {theta.shape=}")
        return None

    X = np.hstack([np.ones((m, 1)), x])
    hx = logistic_predict_(x, theta)

    return (np.dot(X.T, (hx - y)) / m)