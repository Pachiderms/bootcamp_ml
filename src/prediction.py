import numpy as np

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
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        print(f"simple_pred type err: {type(x)=} {type(theta)=}")
        return None
    
    m, n = x.shape
    if theta.shape != (n + 1, 1):
        print(f"simple_pred shape err: {x.shape=} {theta.shape=}")
        return None
    
    y_hat = np.zeros((m, 1))
    for i in range(n):
        y_hat += x[:, i].reshape(-1, 1) * theta[i + 1]
    y_hat += theta[0]
    
    return y_hat

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
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        print(f"pred type err: {type(x)=} {type(theta)=}")
        return None
    
    m, n = x.shape
    if theta.shape != (n + 1, 1):
        print(f"pred shape err: {x.shape=} {theta.shape=}")
        return None
    
    X = np.hstack([np.ones((m, 1)), x])
    y_hat = X @ theta
    
    return y_hat