import numpy as np
from matplotlib import pyplot as plt
from src.log_pred import logistic_predict_
from src.log_gradient import log_gradient
from src.logistic_loss_reg import reg_log_loss_
from src.reg_logistic_grad import vec_reg_logistic_grad
from src.vec_log_loss import vec_log_loss_
from src.decorators import (
    check_type_and_shape_fit_method,
    check_type_and_shape_vector_pair_any,
)


class MyLogisticRegression:
    """
    Description:
    My personnal logistic regression to classify things.My personnal logistic regression to classify things.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000, penality="l2", lambda_=1.0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.asarray(thetas, dtype=float).reshape(-1, 1)
        self.penality = penality if penality == "l2" else None
        self.lambda_ = lambda_ if self.penality == "l2" else 0.0
        self.costs = []

    @check_type_and_shape_fit_method
    def fit_(self, x, y):
        self.costs.clear()

        new_theta = self.thetas.copy()
        for _ in range(self.max_iter):
            y_hat = logistic_predict_(x, new_theta)
            current_loss = self.loss_(y, y_hat, new_theta)
            self.costs.append(current_loss)

            grad = (
                log_gradient(x, y, new_theta)
                if self.penality == None
                else vec_reg_logistic_grad(y, x, new_theta, self.lambda_)
            )
            new_theta = new_theta - self.alpha * grad

        self.thetas = new_theta
        return self

    def plot_learning_curve(self):
        plt.figure()
        plt.xlabel("iterations")
        plt.ylabel("J(w, b)")
        plt.title(f"Loss for lambda={self.lambda_}")

        plt.plot(self.costs, label="learning curve")
        plt.legend()
        plt.show()

    def predict_(self, x):
        return logistic_predict_(x, self.thetas)

    @check_type_and_shape_vector_pair_any
    def loss_elem_(self, y, y_hat):
        J_elem = (y_hat - y) ** 2
        return J_elem

    def loss_(self, y, y_hat, theta):
        return (
            vec_log_loss_(y, y_hat)
            if self.penality == None
            else reg_log_loss_(y, y_hat, theta, self.lambda_)
        )

    @check_type_and_shape_vector_pair_any
    def mse_(self, y, y_hat):
        m = y.shape[0]
        mse = np.sum((y_hat - y) ** 2) / m
        return mse
