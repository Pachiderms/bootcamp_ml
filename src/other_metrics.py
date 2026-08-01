import numpy as np

def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    Returns:
    The accuracy score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
         return None
    if y.shape != y_hat.shape:
        return None
    if len(y) == 0:
        return None

    return float(np.mean(y == y_hat))

def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
    The precision score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
         return None
    if y.shape != y_hat.shape:
        return None
    if len(y) == 0:
        return None
    if pos_label not in y:
        return None

    tp = np.sum((y == pos_label) & (y_hat == pos_label))
    fp = np.sum((y != pos_label) & (y_hat == pos_label))
    return float(tp / (tp + fp))
    
def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
    The recall score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
         return None
    if y.shape != y_hat.shape:
        return None
    if len(y) == 0:
        return None
    if pos_label not in y:
        return None

    tp = np.sum((y == pos_label) & (y_hat == pos_label))
    fn = np.sum((y == pos_label) & (y_hat != pos_label))
    return float(tp / (tp + fn))

def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
    The f1 score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
         return None
    if y.shape != y_hat.shape:
        return None
    if len(y) == 0:
        return None
    if pos_label not in y:
        return None
    
    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)
    return float((2 * precision * recall) / (precision + recall))