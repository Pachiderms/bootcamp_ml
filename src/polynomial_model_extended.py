import numpy as np
from src.decorators import check_type_and_shape_polynomial


@check_type_and_shape_polynomial
def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising its columns to every power in the range
    of 1 up to the power given in argument.
    Args:
    x: has to be an numpy.ndarray, a matrix of shape m * n.
    power: has to be an int, the power up to which the columns of matrix x are going
    to be raised.
    Returns:
    The matrix of polynomial features as a numpy.ndarray, of shape m * (np),
    containg the polynomial feature values for all
    training examples.
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    m, n = x.shape
    P = [np.power(x, pow) for pow in range(1, power + 1)]
    X = np.concatenate(P, axis=1)
    return X
