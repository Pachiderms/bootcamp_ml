import numpy as np

from src.fit import fit_
from src.gradient import gradient, simple_gradient
from src.prediction import predict_


def test_simple_gradient_values():
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape(
        (-1, 1)
    )
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape(
        (-1, 1)
    )

    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    result = simple_gradient(x, y, theta1)
    assert np.allclose(result, np.array([[-19.0342574], [-586.66875564]]))

    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    result = simple_gradient(x, y, theta2)
    assert np.allclose(result, np.array([[-57.86823748], [-2230.12297889]]))


def test_gradient_and_fit():
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape(
        (-1, 1)
    )
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape(
        (-1, 1)
    )

    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    result = gradient(x, y, theta1)
    assert np.allclose(result, np.array([[-19.0342], [-586.6687]]), atol=1e-4)

    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    result = gradient(x, y, theta2)
    assert np.allclose(result, np.array([[-57.8682], [-2230.1229]]), atol=1e-4)

    x_matrix = np.array(
        [[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]]
    )
    y_matrix = np.array(
        [[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]]
    )
    theta = np.array([1, 1]).reshape((-1, 1))
    theta1_fit = fit_(x_matrix, y_matrix, theta, alpha=5e-8, max_iter=1500000)

    assert np.allclose(theta1_fit, np.array([[1.40709365], [1.1150909]]), atol=1e-7)

    prediction = predict_(x_matrix, theta1_fit)
    assert np.allclose(
        prediction,
        np.array(
            [[15.3408728], [25.38243697], [36.59126492], [55.95130097], [65.53471499]]
        ),
        atol=1e-7,
    )
