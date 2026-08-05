import numpy as np
from src.decorators import check_type_and_shape_x_theta


@check_type_and_shape_x_theta
def simple_predict(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
    Return:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not matching.
        None if x or theta is not of expected type.
        Raises:
        This function should not raise any Exception.
    """
    m, n = x.shape
    y_hat = np.zeros((m, 1))
    for i in range(n):
        y_hat += x[:, i].reshape(-1, 1) * theta[i + 1]
    y_hat += theta[0]

    return y_hat


@check_type_and_shape_x_theta
def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimensions m * n.
        theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
    Return:
        y_hat as a numpy.array, a vector of dimensions m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
        None if x or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    m, n = x.shape
    X = np.hstack([np.ones((m, 1)), x])
    y_hat = X @ theta

    return y_hat
