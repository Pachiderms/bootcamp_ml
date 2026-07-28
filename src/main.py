import numpy as np
from matplotlib import pyplot as plt
import sklearn
import pandas as pd
import sklearn

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a one-dimensional array of size m.
    theta: has to be an numpy.ndarray, a one-dimensional array of size 2.
    Returns:
    y_hat as a numpy.ndarray, a one-dimensional array of size m.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """

    m = x.shape[0]
    X = np.column_stack([np.ones(m), x])
    X = X.reshape(m, 2)
    y_hat = X @ theta

    return y_hat

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
        return None
    
    J_elem = (y_hat - y) ** 2
    return J_elem

def loss_simple(y, y_hat):
    """
    Description:
    Calculates the value of loss function.
    Args:
    y: has to be an numpy.array, a two-dimensional array of shape m * 1.
    y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
    Returns:
    J_value : has to be a float.
    None if there is a dimension matching problem.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if y.shape != y_hat.shape:
        return None
    
    m = y.shape[0]
    J_value = 0
    for i in range(m):
        J_value += (y_hat[i][0] - y[i][0]) ** 2
    J_value /= (2 * m)
    return J_value

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
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    if y.shape[0] != y_hat.shape[0]:
        return None

    m = y.shape[0]
    X = np.squeeze(y)
    Y = np.squeeze(y_hat)
    return np.dot((Y - X), (Y - X)) / (2 * m)

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
    if y.shape != y_hat.shape:
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
    if y.shape != y_hat.shape:
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
    if y.shape != y_hat.shape:
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
    
    if y.shape != y_hat.shape:
        return None
    
    ss_res = np.sum((y_hat - y) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2score = 1 - (ss_res / ss_tot)
    
    return r2score

def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.arrays, with a for-loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.array, a vector of shape 2 * 1.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    
    if x.shape != y.shape or theta.shape != (2, 1):
        return None
    
    m = x.shape[0]
    grad = np.zeros((2, 1))
    grad[0] = np.sum(theta[0] + theta[1] * x - y) / m
    grad[1] = np.sum((theta[0] + theta[1] * x - y) * x) / m
    
    return grad

class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.array(thetas)
        
    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) \
            or not isinstance(self.thetas, np.ndarray) or not isinstance(self.alpha, float):
                return None
            
        m, n = x.shape
        if y.shape != (m, 1) or self.thetas.shape != (n + 1, 1):
            return None
    
        new_theta = self.thetas.copy()
        for _ in range(self.max_iter):
            grad = gradient(x, y, new_theta)
            new_theta = new_theta - self.alpha * grad
        
        self.thetas = new_theta
        return self

    def predict_(self, x):
        m = x.shape[0]
        X = np.hstack([np.ones((m, 1)), x])
        y_hat = X @ self.thetas

        return y_hat
    
    def loss_elem_(self, y, y_hat):
        if y.shape[0] != y_hat.shape[0]:
            return None
        
        J_elem = (y_hat - y) ** 2
        return J_elem
    
    def loss_(self, y, y_hat):
        if y.shape[0] != y_hat.shape[0]:
            return None
        
        m = y.shape[0]
        return np.sum((y_hat - y) ** 2) / (2 * m)
    
    def mse_(self, y, y_hat):
        if y.shape != y_hat.shape:
            return None
        
        m = y.shape[0]
        mse = np.sum((y_hat - y) ** 2) / m
        return mse

def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x’ as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn’t raise any Exception.
    """
    
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    
    m = x.shape[0]
    mean = 0
    for i in range(m):
        mean += x[i]
        
    mean = mean / m if m > 0 else 0
        
    std = 0
    for i in range(m):
        std += (x[i] - mean) ** 2
    std /= m if m > 0 else 0
    std = np.sqrt(std)
    
    x_normalized = (x - mean) / np.where(std == 0, 1, std)
    
    return x_normalized

def minmax(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x’ as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn’t raise any Exception.
    """
    
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    
    min_val = np.min(x)
    max_val = np.max(x)
    
    if max_val == min_val:
        return np.zeros(x.shape)
    
    x_normalized = (x - min_val) / (max_val - min_val)
    
    return x_normalized

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
        return None
    
    m, n = x.shape
    if theta.shape != (n + 1, 1):
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
        return None
    
    m, n = x.shape
    if theta.shape != (n + 1, 1):
        return None
    
    X = np.hstack([np.ones((m, 1)), x])
    y_hat = X @ theta
    
    return y_hat

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
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    if y.shape[0] != y_hat.shape[0]:
        return None
    
    m = y.shape[0]
    mse = np.sum((y_hat - y) ** 2) / (2 * m)
    return mse

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
        The gradient as a numpy.array, a vector of dimensions n * 1,
        containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible dimensions.
        None if x, y or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
        
    m, n = x.shape
    if y.shape != (m, 1) or theta.shape != (n + 1, 1):
        return None

    X = np.hstack((np.ones((m, 1)), x))
    
    errors = X @ theta - y
    grad = X.T @ errors / m
    
    return grad

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a matrix of dimension m * n:
        (number of training examples, number of features).
        y: has to be a numpy.array, a vector of dimension m * 1:
        (number of training examples, 1).
        theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
        (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
        new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) \
        or not isinstance(theta, np.ndarray) or not isinstance(alpha, float):
        return None
    
    m, n = x.shape
    if y.shape != (m, 1) or theta.shape != (n + 1, 1):
        return None

    new_theta = theta.copy()
    for _ in range(max_iter):
        grad = gradient(x, y, new_theta)
        new_theta = new_theta - alpha * grad

    return new_theta

class MyLinearRegression2():
    """
    Description:
    My personnal multiple linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        """_summary_

        Args:
            thetas (np.array): _description_
            alpha (float, optional): _description_. Defaults to 0.001.
            max_iter (int, optional): _description_. Defaults to 1000.
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        
    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            print("type err")
            return None
        
        m, n = x.shape
        if y.shape != (m, 1) or self.thetas.shape != (n + 1, 1):
            print("shape err")
            return None
        
        plt.figure()
        plt.xlabel("iterations")
        plt.ylabel("J(w, b)")
                
        new_theta = self.thetas.copy()
        costs = []
        # colors = plt.cm.rainbow(np.linspace(0, 1, self.max_iter))
        
        for i in range(self.max_iter):
            y_hat = predict_(x, new_theta)
            current_loss = self.loss_(y, y_hat)
            costs.append(current_loss)
            
            grad = gradient(x, y, new_theta)
            new_theta = new_theta - self.alpha * grad
            
        plt.plot(costs, label='learning curve')
        plt.legend()
        plt.show()

        self.thetas = new_theta
        return self

    def predict_(self, x):
        if not isinstance(x, np.ndarray):
            print("x err")
            return None
        
        m, n = x.shape
        
        if self.thetas.shape != (n + 1, 1):
            print("theta err")
            print(f"x shape: {x.shape}")
            print(self.thetas.shape)
            return None
        
        X = np.hstack([np.ones((m, 1)), x])
        y_hat = X @ self.thetas
        
        return y_hat
    
    def loss_elem_(self, y, y_hat):
        if y.shape[0] != y_hat.shape[0]:
            return None
        
        J_elem = (y_hat - y) ** 2
        return J_elem
    
    def loss_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        
        if y.shape[0] != y_hat.shape[0]:
            return None
        
        m = y.shape[0]
        loss = np.sum((y_hat - y) ** 2) / (2 * m)
        return loss
    
    def mse_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        
        if y.shape != y_hat.shape:
            return None
        
        m = y.shape[0]
        mse = np.sum((y_hat - y) ** 2) / m
        return mse

class MyPloter():
    def __init__(self, sp=(1, 1), fs=(20, 5)):
        self.fig, self.ax = plt.subplots(sp[0], sp[1], figsize=fs)
    
    def scatter(self, x, y, xlabel="x", ylabel="y", color="blue", label="Data", s=20, ax_id=0):    
        ax = self.ax[ax_id] if isinstance(self.ax, np.ndarray) else self.ax
    
        ax.scatter(x, y, color=color, label=label, s=s)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        
    def plot(self, x, y, param_dict={}, ax_id=0):
        ax = self.ax[ax_id] if isinstance(self.ax, np.ndarray) else self.ax
        
        ax.plot(x, y, **param_dict)
        ax.legend()

def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    power: has to be an int, the power up to which the components of vector x are going to be raised.
    Return:
    The matrix of polynomial features as a numpy.array, of dimension m * n,
    containing the polynomial feature values for all training examples.
    None if x is an empty numpy.array.
    None if x or power is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    
    if not isinstance(x, np.ndarray) or not isinstance(power, int):
        return None
    
    if x.shape[1] != 1:
        return None
    
    e = np.arange(1, power + 1)
    
    return(x ** e)
