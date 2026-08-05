import numpy as np
from src.decorators import check_type_and_shape_vector_pair_any


@check_type_and_shape_vector_pair_any
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
    return float(np.mean(y == y_hat))


@check_type_and_shape_vector_pair_any
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
    if pos_label not in y:
        print(f"precision label err: {pos_label=} {y=}")
        return None

    tp = np.sum((y == pos_label) & (y_hat == pos_label))
    fp = np.sum((y != pos_label) & (y_hat == pos_label))
    return float(tp / (tp + fp))


@check_type_and_shape_vector_pair_any
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
    if pos_label not in y:
        print(f"recall label err: {pos_label=} {y=}")
        return None

    tp = np.sum((y == pos_label) & (y_hat == pos_label))
    fn = np.sum((y == pos_label) & (y_hat != pos_label))
    return float(tp / (tp + fn))


@check_type_and_shape_vector_pair_any
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
    if pos_label not in y:
        print(f"f1 label err: {pos_label=} {y=}")
        return None

    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)
    return float((2 * precision * recall) / (precision + recall))
