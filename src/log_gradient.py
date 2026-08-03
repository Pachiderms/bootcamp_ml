import numpy as np
from src.log_pred import logistic_predict_
from src.decorators import check_type_and_shape_xy_theta

@check_type_and_shape_xy_theta
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
    m, n = x.shape
    X = np.hstack([np.ones((m, 1)), x])
    hx = logistic_predict_(x, theta)

    return (np.dot(X.T, (hx - y)) / m)