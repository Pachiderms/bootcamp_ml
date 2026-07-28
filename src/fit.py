import numpy as np
from src.gradient import gradient

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a matrix of dimension m * n:
        (number of training examples, number of features).
        y: has to be a numpy.array, a vector of dimension m * 1:
        (number of training examples, 1).
        theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
        (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
        new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) \
        or not isinstance(theta, np.ndarray) or not isinstance(alpha, float):
        return None
    
    m, n = x.shape
    if y.shape != (m, 1) or theta.shape != (n + 1, 1):
        return None

    new_theta = theta.copy()
    for _ in range(max_iter):
        grad = gradient(x, y, new_theta)
        new_theta = new_theta - alpha * grad

    return new_theta