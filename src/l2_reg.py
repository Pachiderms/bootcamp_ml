import numpy as np

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
    if not isinstance(theta, np.ndarray):
        print(f"it_l2 type err: {type(theta)=}")
        return None
    if theta.shape[1] != 1:
        print(f"it_l2 shape err: {theta.shape=}")
        return None
    
    T = theta.copy()
    T[0] = 0
    sum = 0
    for _, t in enumerate(T):
        sum = sum + t ** 2
    return sum.item()
    
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
    if not isinstance(theta, np.ndarray):
            print(f"l2 type err: {type(theta)=}")
            return None
    if theta.shape[1] != 1:
        print(f"l2 shape err: {theta.shape=}")
        return None
        
    T = theta.copy()
    T[0] = 0
    return (T.T @ T).item()
