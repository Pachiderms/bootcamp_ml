import numpy as np

def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of the dataset that will be assigned to the
        training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible dimensions.
        None if x, y or proportion is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        print(f"split type err: {type(x)=} {type(y)=}")
        return None

    m = x.shape[0]
    if y.shape != (m, 1):
        print(f"err {y.shape=} {x.shape=}")
        return None
    
    np.random.seed(0)
    np.random.shuffle(x)
    np.random.seed(0)
    np.random.shuffle(y)

    return (x[:int(m * proportion)], x[int(m * proportion):],
            y[:int(m * proportion)], y[int(m * proportion):])
