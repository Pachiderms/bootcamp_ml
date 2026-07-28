from matplotlib import pyplot as plt
from src.prediction import predict_

def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a one-dimensional array of size m.
    y: has to be an numpy.array, a one-dimensional array of size m.
    theta: has to be an numpy.array, a two-dimensional array of shape 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    y_hat = predict_(x, theta)
    fig = plt.figure()
    plt.plot(x, y, marker='o')
    plt.plot(x, y_hat, color='orange')
    plt.show()
    
def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, one-dimensional array of size m.
    y: has to be an numpy.ndarray, one-dimensional array of size m.
    theta: has to be an numpy.ndarray, one-dimensional array of size 2.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    
    if x.shape[0] != 1 or y.shape[0] != 1 or theta.size != 2:
        return None

    m = x.shape[0]
    print(m)
    y_hat = predict_(x, theta)
    fig = plt.figure()
    plt.plot(x, y, marker='o')
    plt.plot(x, y_hat, color='orange')
    for i in range(m):
        plt.plot([x[i], x[i]], [y[i], y_hat[i]], linestyle='dashed', color='red')
    plt.show()