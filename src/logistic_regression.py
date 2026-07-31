import numpy as np
from matplotlib import pyplot as plt
from src.log_pred import logistic_predict_
from src.log_gradient import log_gradient
from src.vec_log_loss import vec_log_loss_

class MyLogisticRegression():
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
            y_hat = logistic_predict_(x, new_theta)
            current_loss = self.loss_(y, y_hat)
            self.costs.append(current_loss)
                            
            grad = log_gradient(x, y, new_theta)
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
        return logistic_predict_(x, self.thetas)
    
    def loss_elem_(self, y, y_hat):
        if y.shape[0] != y_hat.shape[0]:
            return None
        
        J_elem = (y_hat - y) ** 2
        return J_elem
    
    def loss_(self, y, y_hat, eps=1e-15):
        return vec_log_loss_(y, y_hat)    

    def mse_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        
        if y.shape != y_hat.shape:
            return None
        
        m = y.shape[0]
        mse = np.sum((y_hat - y) ** 2) / m
        return mse