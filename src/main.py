import numpy as np
from matplotlib import pyplot as plt
from src.prediction import predict_
from src.gradient import gradient


class MyLinearRegression():
    """
    Description:
    My personnal multiple linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.asarray(thetas, dtype=float).reshape(-1, 1)
        self.costs = []
        
    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) \
            or not isinstance(self.thetas, np.ndarray) or not isinstance(self.alpha, float):
            print('type err')
            return None
            
        m, n = x.shape
        if y.shape != (m, 1) or self.thetas.shape != (n + 1, 1):
            print(f'shape err: {x.shape=} {y.shape=} {self.thetas.shape=}')
            return None

        self.costs.clear()
        
        new_theta = self.thetas.copy()
        for _ in range(self.max_iter):
            y_hat = predict_(x, new_theta)
            current_loss = self.loss_(y, y_hat)
            self.costs.append(current_loss)
                            
            grad = gradient(x, y, new_theta)
            new_theta = new_theta - self.alpha * grad
            
        self.thetas = new_theta
        return self

    def plot_learning_curve(self):
        plt.figure()
        plt.xlabel("iterations")
        plt.ylabel("J(w, b)")

        plt.plot(self.costs, label='learning curve')
        plt.legend()
        plt.show()

    def predict_(self, x):
        if not isinstance(x, np.ndarray):
            return None
        
        m, n = x.shape
        
        if self.thetas.shape != (n + 1, 1):
            print('theta shape err')
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

def add_polynomial_features_mult(x, degree):
    return np.hstack([
        add_polynomial_features(x[:, [i]], degree)
        for i in range(x.shape[1])
    ])

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
