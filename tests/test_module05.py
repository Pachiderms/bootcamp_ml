import numpy as np

from src.fit import fit_
from src.gradient import gradient
from src.loss import loss_
from src.plot import plot, plot_with_loss
from src.prediction import predict_


def test_predict_basic_examples():
    x = np.arange(1, 13).reshape((4, -1))

    theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
    assert np.allclose(predict_(x, theta1), np.array([[5.0], [5.0], [5.0], [5.0]]))

    theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
    assert np.allclose(predict_(x, theta2), np.array([[1.0], [4.0], [7.0], [10.0]]))

    theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
    assert np.allclose(
        predict_(x, theta3), np.array([[9.64], [24.28], [38.92], [53.56]])
    )

    theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
    assert np.allclose(predict_(x, theta4), np.array([[12.5], [32.0], [51.5], [71.0]]))


def test_loss_and_gradient_examples():
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    assert loss_(X, Y) == 2.142857142857143
    assert loss_(X, X) == 0.0

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
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    theta1 = np.array([0, 3, 0.5, -6]).reshape((-1, 1))
    assert np.allclose(
        gradient(x, y, theta1),
        np.array([[-33.71428571], [-37.35714286], [183.14285714], [-393.0]]),
    )

    theta2 = np.array([0, 0, 0, 0]).reshape((-1, 1))
    assert np.allclose(
        gradient(x, y, theta2),
        np.array([[-0.71428571], [0.85714286], [23.28571429], [-26.42857143]]),
    )


def test_fit_finds_expected_coefficients():
    x = np.array(
        [[0.2, 2.0, 20.0], [0.4, 4.0, 40.0], [0.6, 6.0, 60.0], [0.8, 8.0, 80.0]]
    )
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.0], [1.0], [1.0], [1.0]])
    theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42000)

    assert np.allclose(
        theta2, np.array([[41.9999], [0.9714], [0.7714], [-1.2000]]), rtol=1e-1
    )
    assert np.allclose(
        predict_(x, theta2),
        np.array([[19.5992], [-2.8003], [-25.1999], [-47.5996]]),
        rtol=1e-2,
    )


def test_plot_helpers_do_not_crash():
    x = np.arange(1, 6)
    y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
    theta1 = np.array([18, -1])
    theta2 = np.array([14, 0])
    theta3 = np.array([12, 0.8])

    plot(x, y, theta1)
    plot_with_loss(x, y, theta1)
    plot(x, y, theta2)
    plot_with_loss(x, y, theta2)
    plot(x, y, theta3)
    plot_with_loss(x, y, theta3)
