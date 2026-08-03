import numpy as np
from src.decorators import check_type_and_shape_theta

@check_type_and_shape_theta
def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    T = theta.copy()
    T[0] = 0
    sum = 0
    for _, t in enumerate(T):
        sum = sum + t ** 2
    return sum.item()
    
@check_type_and_shape_theta
def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    T = theta.copy()
    T[0] = 0
    return (T.T @ T).item()
