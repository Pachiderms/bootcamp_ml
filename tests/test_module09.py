import math

import numpy as np

from src.benchmark_train import train_models_reg
from src.l2_reg import iterative_l2, l2
from src.linear_loss_reg import reg_loss_
from src.polynomial_model_extended import add_polynomial_features
from src.logistic_loss_reg import reg_log_loss_
from src.reg_linear_grad import reg_linear_grad, vec_reg_linear_grad
from src.reg_logistic_grad import reg_logistic_grad, vec_reg_logistic_grad
import numpy as np
from src.logistic_regression import MyLogisticRegression as mylogr
from src.space_avocado import split_set
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def test_polynomial_and_l2_regularization():
    x = np.arange(1, 11).reshape(5, 2)
    assert np.allclose(
        add_polynomial_features(x, 3),
        np.array(
            [
                [1, 2, 1, 4, 1, 8],
                [3, 4, 9, 16, 27, 64],
                [5, 6, 25, 36, 125, 216],
                [7, 8, 49, 64, 343, 512],
                [9, 10, 81, 100, 729, 1000],
            ]
        ),
    )
    assert np.allclose(
        add_polynomial_features(x, 4),
        np.array(
            [
                [1, 2, 1, 4, 1, 8, 1, 16],
                [3, 4, 9, 16, 27, 64, 81, 256],
                [5, 6, 25, 36, 125, 216, 625, 1296],
                [7, 8, 49, 64, 343, 512, 2401, 4096],
                [9, 10, 81, 100, 729, 1000, 6561, 10000],
            ]
        ),
    )

    x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    assert iterative_l2(x) == 911.0
    assert l2(x) == 911.0

    y = np.array([3, 0.5, -6]).reshape((-1, 1))
    assert iterative_l2(y) == 36.25
    assert l2(y) == 36.25


def test_regularized_losses_and_gradients():
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
    assert reg_loss_(y, y_hat, theta, 0.5) == 0.8503571428571429
    assert reg_loss_(y, y_hat, theta, 0.05) == 0.5511071428571429
    assert reg_loss_(y, y_hat, theta, 0.9) == 1.116357142857143

    y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
    y_hat = np.array([0.9, 0.79, 0.12, 0.04, 0.89, 0.93, 0.01]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
    assert math.isclose(
        reg_log_loss_(y, y_hat, theta, 0.5), 0.43377043716475955, rel_tol=1e-14
    )
    assert math.isclose(reg_log_loss_(y, y_hat, theta, 0.05), 0.13452043716475953)
    assert math.isclose(reg_log_loss_(y, y_hat, theta, 0.9), 0.6997704371647596)

    x = np.array(
        [
            [-6, -7, -9],
            [13, -2, 14],
            [-7, 14, -1],
            [-8, -4, 6],
            [-5, -9, 6],
            [1, -5, 11],
            [9, -11, 8],
        ]
    )
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])
    assert np.allclose(
        reg_linear_grad(y, x, theta, 1.0),
        np.array([[-60.99], [-195.64714286], [863.46571429], [-644.52142857]]),
    )
    assert np.allclose(
        vec_reg_linear_grad(y, x, theta, 1.0),
        np.array([[-60.99], [-195.64714286], [863.46571429], [-644.52142857]]),
    )

    x = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    assert np.allclose(
        reg_logistic_grad(y, x, theta, 1.0),
        np.array(
            [[-0.55711039], [-1.40334809], [-1.91756886], [-2.56737958], [-3.03924017]]
        ),
    )
    assert np.allclose(
        vec_reg_logistic_grad(y, x, theta, 1.0),
        np.array(
            [[-0.55711039], [-1.40334809], [-1.91756886], [-2.56737958], [-3.03924017]]
        ),
    )


def test_ridge_pipeline_and_model_training():
    x_train, x_test, y_train, y_test = split_set()
    models, errors, predictions = train_models_reg(x_train, x_test, y_train, y_test, 4)

    best_index = int(np.argmin(errors))
    assert 0 <= best_index < len(models)
    assert np.isfinite(errors[best_index])
    assert predictions[best_index].shape[0] == len(x_test)

    for model in models:
        model.plot_learning_curve()

    print(f"{errors=}")

    weight = x_test[:, 0]
    prod_distance = x_test[:, 1]
    time_delivery = x_test[:, 2]
    y_hat = predictions[0]

    plt.figure(figsize=(18, 6))

    plt.scatter(weight, time_delivery, c=y_test)
    plt.scatter(weight, time_delivery, c=y_hat, marker="x", s=40, alpha=0.4)

    plt.scatter(weight, prod_distance, c=y_test)
    plt.scatter(weight, prod_distance, c=y_hat, marker="x", s=40, alpha=0.4)

    plt.scatter(time_delivery, prod_distance, c=y_test)
    plt.scatter(time_delivery, prod_distance, c=y_hat, marker="x", s=40, alpha=0.4)

    print(best_index)
    best_model = models[best_index]
    print(best_model.get_params_())

    xtst = add_polynomial_features(x_test, 2)
    y_hat = best_model.predict_(xtst)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].scatter(y_test, y_hat, alpha=0.5)
    ax[0].plot([y_hat.min(), y_hat.max()], [y_hat.min(), y_hat.max()], "r--")
    ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "y--")

    ax[0].set_xlabel("True values")
    ax[0].set_ylabel("Predicted values")
    ax[0].set_title("Predicted vs True")

    ax[1].plot(y_test, label="True")
    ax[1].plot(y_hat, label="Predicted")
    ax[1].legend()
    plt.show()


def test_logistic_regression_regularized():
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    # Example 1:
    model1 = mylogr(theta, lambda_=5.0)
    assert model1.penality == "l2"
    assert model1.lambda_ == 5.0
    # Example 2:
    model2 = mylogr(theta, penality=None)
    assert model2.penality == None
    assert model2.lambda_ == 0.0
    # Example 3:
    model3 = mylogr(theta, penality=None, lambda_=2.0)
    assert model3.penality == None
    assert model3.lambda_ == 0.0
