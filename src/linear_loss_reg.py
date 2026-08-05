import numpy as np
from src.l2_reg import l2
from src.decorators import check_type_and_shape_reg_loss


@check_type_and_shape_reg_loss
def reg_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a linear regression model from two non-empty numpy.array,
    without any for loop. The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    The regularized loss as a float.
    None if y, y_hat, or theta are empty numpy.ndarray.
    None if y and y_hat do not share the same shapes.
    Raises:
    This function should not raise any Exception.
    """
    m = y.shape[0]
    J = np.dot((y_hat - y).T, (y_hat - y)) + (lambda_ * l2(theta))
    return J.item() / (2 * m)
