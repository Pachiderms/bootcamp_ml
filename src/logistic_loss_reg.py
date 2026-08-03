import numpy as np
from src.l2_reg import l2
from src.decorators import check_type_and_shape_reg_loss

@check_type_and_shape_reg_loss
def reg_log_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray,
    without any for loop. The two arrays must have the same shapes.
    Args:
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    The regularized loss as a float.
    None if y, y_hat, or theta is empty numpy.ndarray.
    None if y and y_hat do not share the same shapes.
    Raises:
    This function should not raise any Exception.
    """
    m = y.shape[0]
    return ((-1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)).item()) + ((lambda_ / (2 * m)) * l2(theta))
    