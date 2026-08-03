import numpy as np

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
    
    if not isinstance(x, np.ndarray) or not isinstance(power, int):
        print(f"qdd_p_features type err: {type(x)} {type(power)}")
        return None
    
    m, n = x.shape
    P = [np.power(x, pow) for pow in range(1, power + 1)]
    X = np.concatenate(P,  axis=1)
    return(X)
