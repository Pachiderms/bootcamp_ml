import numpy as np
from src.prediction import predict_
from src.decorators import check_type_and_shape_reg_gradient

@check_type_and_shape_reg_gradient
def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray,
    with two for-loop. The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    m, n = x.shape
    grad = np.zeros((n + 1, 1))
    hx = predict_(x, theta)

    for i in range(m):
        error = (hx[i, 0] - y[i, 0]).item()

        grad[0, 0] += error

        for j in range(n):
            grad[j + 1, 0] += error * x[i, j]
    
    for j in range(1, n + 1):
                grad[j, 0] += lambda_ * theta[j, 0]

    grad /= m

    return grad


@check_type_and_shape_reg_gradient
def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    m = x.shape[0]
    X = np.hstack((np.ones((m, 1)), x))
        
    hx = X @ theta
    Theta = theta.copy()
    Theta[0] = 0
    grad = (X.T @ (hx - y) +  + (lambda_ * Theta)) / m
        
    return grad
