import numpy as np

def loss_elem_(y, y_hat):
    """
    Description:
    Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
    y: has to be an numpy.array, a two-dimensional array of shape m * 1.
    y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
    Returns:
    J_elem: numpy.array, a array of dimension (number of the training examples, 1).
    None if there is a dimension matching problem.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if y.shape != y_hat.shape:
        print(f"loss_elem shape err: {y.shape=} {y_hat.shape=}")
        return None
    
    J_elem = (y_hat - y) ** 2
    return J_elem

def loss_simple(y, y_hat):
    """Computes the mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Return:
        The mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
        None if y or y_hat is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        print(f"loss_simple type err: {type(y)=} {type(y_hat)=}")
        return None
    if y.size == 0 or y_hat.size == 0:
        print(f"loss_simple empty err: {y.size=} {y_hat.size=}")
        return None
    if y.shape[0] != y_hat.shape[0]:
        print(f"loss_simple shape err: {y.shape=} {y_hat.shape=}")
        return None
    
    m = y.shape[0]
    mse = np.sum((y_hat - y) ** 2) / (2 * m)
    return mse

def loss_(y, y_hat):
    """
    Computes the mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Return:
        The mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
        None if y or y_hat is not of expected type.
    Raises:
        This function should not raise any Exception.
    """

    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        print(f"loss type err: {type(y)=} {type(y_hat)=}")
        return None
    if y.size == 0 or y_hat.size == 0:
        print(f"loss empty err: {y.size=} {y_hat.size=}")
        return None
    if y.shape[0] != y_hat.shape[0]:
        print(f"loss shape err: {y.shape=} {y_hat.shape=}")
        return None

    m = y.shape[0]
    X = np.squeeze(y)
    Y = np.squeeze(y_hat)
    return np.dot((Y - X), (Y - X)) / (2 * m)

def mse_(y, y_hat):
    """
    Description:
    Calculate the MSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a two-dimensional array of shape m * 1.
    y_hat: has to be a numpy.array, a two-dimensional vector of shape m * 1.
    Returns:
    mse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        print(f"mse type err: {type(y)=} {type(y_hat)=}")
        return None
    if y.shape != y_hat.shape:
        print(f"mse shape err: {y.shape=} {y_hat.shape=}")
        return None
        
    m = y.shape[0]
    mse = np.sum((y_hat - y) ** 2) / m
    return mse

def rmse_(y, y_hat):
    """
    Description:
    Calculate the RMSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a two-dimensional array of shape m * 1.
    y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
    Returns:
    rmse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        print(f"rmse type err: {type(y)=} {type(y_hat)=}")
        return None
    if y.shape != y_hat.shape:
        print(f"rmse shape err: {y.shape=} {y_hat.shape=}")
        return None

    m = y.shape[0]
    rmse = np.sqrt(np.sum((y_hat - y) ** 2) / m)
    return rmse

def mae_(y, y_hat):
    """
    Description:
    Calculate the MAE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a two-dimensional array of shape m * 1.
    y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
    Returns:
    mae: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        print(f"mae type err: {type(y)=} {type(y_hat)=}")
        return None
    if y.shape != y_hat.shape:
        print(f"mae shape err: {y.shape=} {y_hat.shape=}")
        return None

    m = y.shape[0]
    mae = np.sum(np.abs(y_hat - y)) / m
    return mae


def r2score_(y, y_hat):
    """
    Description:
    Calculate the R2score between the predicted output and the output.
    Args:
    y: has to be a numpy.array, a two-dimensional array of shape m * 1.
    y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
    Returns:
    r2score: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        print(f"r2score type err: {type(y)=} {type(y_hat)=}")
        return None
    if y.shape != y_hat.shape:
        print(f"r2score shape err: {y.shape=} {y_hat.shape=}")
        return None
    
    ss_res = np.sum((y_hat - y) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2score = 1 - (ss_res / ss_tot)
    
    return r2score