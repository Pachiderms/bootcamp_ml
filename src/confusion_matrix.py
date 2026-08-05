import numpy as np
from src.decorators import check_type_and_shape_confusion


@check_type_and_shape_confusion
def confusion_matrix_(y_true, y_hat, labels=None):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
    y_true: numpy.ndarray for the correct labels
    y_hat: numpy.ndarray for the predicted labels
    labels: Optional, a list of labels to index the matrix.
    This may be used to reorder or select a subset of labels. (default=None)
    Returns:
    The confusion matrix as a numpy ndarray.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    l = np.append(y_true, y_hat).reshape(-1, 1)
    labels = labels or l
    u_labels = np.unique_counts(labels).values.reshape(-1, 1)
    m, _ = u_labels.shape
    cf_m = np.zeros((m, m))

    for i, label_1 in enumerate(u_labels):
        for j, label_2 in enumerate(u_labels):
            true = (y_true == label_1).astype(int)
            pred = (y_hat == label_2).astype(int)
            cf_m[i, j] = np.sum((true == pred) & (true == 1))

    return cf_m
