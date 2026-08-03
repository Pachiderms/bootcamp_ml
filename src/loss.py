import numpy as np
from src.decorators import check_type_and_shape_vector_pair_any

@check_type_and_shape_vector_pair_any
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
    J_elem = (y_hat - y) ** 2
    return J_elem

@check_type_and_shape_vector_pair_any
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
    m = y.shape[0]
    mse = np.sum((y_hat - y) ** 2) / (2 * m)
    return mse

@check_type_and_shape_vector_pair_any
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

    m = y.shape[0]
    X = np.squeeze(y)
    Y = np.squeeze(y_hat)
    return np.dot((Y - X), (Y - X)) / (2 * m)

@check_type_and_shape_vector_pair_any
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
    m = y.shape[0]
    mse = np.sum((y_hat - y) ** 2) / m
    return mse

@check_type_and_shape_vector_pair_any
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
    m = y.shape[0]
    rmse = np.sqrt(np.sum((y_hat - y) ** 2) / m)
    return rmse

@check_type_and_shape_vector_pair_any
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
    m = y.shape[0]
    mae = np.sum(np.abs(y_hat - y)) / m
    return mae


@check_type_and_shape_vector_pair_any
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
    ss_res = np.sum((y_hat - y) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2score = 1 - (ss_res / ss_tot)
    
    return r2score