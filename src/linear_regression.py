import numpy as np
from matplotlib import pyplot as plt
from src.prediction import predict_
from src.gradient import gradient
from src.loss import loss_elem_, loss_, mse_
from src.decorators import (
    check_type_and_shape_fit_method,
    check_type_and_shape_polynomial,
)


class MyLinearRegression:
    """
    Description:
    My personnal multiple linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.asarray(thetas, dtype=float).reshape(-1, 1)
        self.costs = []

    @check_type_and_shape_fit_method
    def fit_(self, x, y):
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

        plt.plot(self.costs, label="learning curve")
        plt.legend()
        plt.show()

    def predict_(self, x):
        return predict_(x, self.thetas)

    def loss_elem_(self, y, y_hat):
        return loss_elem_(y, y_hat)

    def loss_(self, y, y_hat):
        return loss_(y, y_hat)

    def mse_(self, y, y_hat):
        return mse_(y, y_hat)


def add_polynomial_features_mult(x, degree):
    return np.hstack(
        [add_polynomial_features(x[:, [i]], degree) for i in range(x.shape[1])]
    )


@check_type_and_shape_polynomial
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
    e = np.arange(1, power + 1)

    return x**e
