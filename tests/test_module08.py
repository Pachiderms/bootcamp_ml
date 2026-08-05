import math

import numpy as np

from src.log_gradient import log_gradient
from src.log_loss import log_loss_
from src.log_pred import logistic_predict_
from src.logistic_regression import MyLogisticRegression as MyLR
from src.sigmoid import sigmoid_
from src.vec_log_loss import vec_log_loss_


def test_sigmoid_values():
    x = np.linspace(-10, 10).reshape(-1, 1)
    assert sigmoid_(x).shape == x.shape

    x = np.array([[-4]])
    assert np.allclose(sigmoid_(x), np.array([[0.01798620996209156]]))

    x = np.array([[2]])
    assert np.allclose(sigmoid_(x), np.array([[0.8807970779778823]]))

    x = np.array([[-4], [2], [0]])
    assert np.allclose(
        sigmoid_(x), np.array([[0.01798620996209156], [0.8807970779778823], [0.5]])
    )


def test_logistic_predict_and_losses():
    x = np.array([4]).reshape((-1, 1))
    theta = np.array([[2], [0.5]])
    assert np.allclose(logistic_predict_(x, theta), np.array([[0.98201379]]))

    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    assert np.allclose(
        logistic_predict_(x2, theta2),
        np.array(
            [[0.98201379], [0.99624161], [0.97340301], [0.99875204], [0.90720705]]
        ),
    )

    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    y_hat1 = logistic_predict_(x1, theta1)
    assert np.allclose(log_loss_(y1, y_hat1), 0.01814992791780973)
    assert np.allclose(vec_log_loss_(y1, y_hat1), 0.01814992791780973)

    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    y_hat2 = logistic_predict_(x2, theta2)
    assert np.allclose(log_loss_(y2, y_hat2), 2.4825011602474483)
    assert np.allclose(vec_log_loss_(y2, y_hat2), 2.4825011602474483)


def test_log_gradient_and_training():
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    assert np.allclose(
        log_gradient(x1, y1, theta1), np.array([[-0.01798621], [-0.07194484]])
    )

    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    assert np.allclose(
        log_gradient(x2, y2, theta2), np.array([[0.3715235], [3.25647547]])
    )

    X = np.array([[1.0, 1.0, 2.0, 3.0], [5.0, 8.0, 13.0, 21.0], [3.0, 5.0, 9.0, 14.0]])
    Y = np.array([[1], [0], [1]])
    thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
    model = MyLR(thetas, penality=None, alpha=1e-5, max_iter=100000)

    y_hat = model.predict_(X)
    assert np.allclose(y_hat, np.array([[0.99930437], [1.0], [1.0]]))
    assert math.isclose(model.loss_(Y, y_hat), 11.513157421577002, rel_tol=1e-4)

    model.fit_(X, Y)
    assert np.allclose(
        model.thetas,
        np.array(
            [[2.11826435], [0.10154334], [6.43942899], [-5.10817488], [0.6212541]]
        ),
        rtol=1e-3,
    ), f"{model.thetas=}"

    y_hat2 = model.predict_(X)
    assert np.allclose(
        y_hat2, np.array([[0.57606717], [0.68599807], [0.06562156]]), rtol=1e-3
    )
    assert math.isclose(model.loss_(Y, y_hat2), 1.4779126923052268, rel_tol=1e-4)
