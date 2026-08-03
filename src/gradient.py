import numpy as np

def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.arrays, with a for-loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.array, a vector of shape 2 * 1.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        print(f"simple_grad type err: {type(x)=} {type(y)=} {type(theta)=}")
        return None

    if x.shape != y.shape or theta.shape != (2, 1):
        print(f"simple_grad shape err: {x.shape=} {y.shape=} {theta.shape=}")
        return None
    
    m = x.shape[0]
    grad = np.zeros((2, 1))
    grad[0] = np.sum(theta[0] + theta[1] * x - y) / m
    grad[1] = np.sum((theta[0] + theta[1] * x - y) * x) / m
    
    return grad

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
        The gradient as a numpy.array, a vector of dimensions n * 1,
        containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible dimensions.
        None if x, y or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        print(f"grad type err: {type(x)=} {type(y)=} {type(theta)=}")
        return None
        
    m, n = x.shape
    if y.shape != (m, 1) or theta.shape != (n + 1, 1):
        print(f"grad shape err: {x.shape=} {y.shape=} {theta.shape=}")
        return None

    X = np.hstack((np.ones((m, 1)), x))
    
    errors = X @ theta - y
    grad = X.T @ errors / m
    
    return grad
