import numpy as np
from src.sigmoid import sigmoid_
from src.decorators import check_type_and_shape_x_theta


@check_type_and_shape_x_theta
def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    m, n = x.shape
    X = np.hstack([np.ones((m, 1)), x])
    vec = np.dot(X, theta)
    return sigmoid_(vec)
