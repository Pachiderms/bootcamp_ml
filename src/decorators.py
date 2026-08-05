import functools
import numpy as np


def _debug_print(func_name, error_type, **values):
    details = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"{func_name} {error_type} err: {details}")


def check_type_and_shape_reg_gradient(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        y, x, theta, lambda_ = args
        if (
            not isinstance(y, np.ndarray)
            or not isinstance(x, np.ndarray)
            or not isinstance(theta, np.ndarray)
            or not isinstance(lambda_, float)
        ):
            _debug_print(
                func.__name__,
                "type",
                y=type(y),
                x=type(x),
                theta=type(theta),
                lambda_=type(lambda_),
            )
            return None
        m, n = x.shape
        if y.shape != (m, 1) or theta.shape != (n + 1, 1):
            _debug_print(
                func.__name__, "shape", x=x.shape, y=y.shape, theta=theta.shape
            )
            return None
        return func(y, x, theta, lambda_)

    return wrapper


def check_type_and_shape_xy_theta(func):
    @functools.wraps(func)
    def wrapper(x, y, theta, *args, **kwargs):
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(y, np.ndarray)
            or not isinstance(theta, np.ndarray)
        ):
            _debug_print(func.__name__, "type", x=type(x), y=type(y), theta=type(theta))
            return None
        m, n = x.shape
        if y.shape != (m, 1) or theta.shape != (n + 1, 1):
            _debug_print(
                func.__name__, "shape", x=x.shape, y=y.shape, theta=theta.shape
            )
            return None
        return func(x, y, theta, *args, **kwargs)

    return wrapper


def check_type_and_shape_x_theta(func):
    @functools.wraps(func)
    def wrapper(x, theta, *args, **kwargs):
        if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
            _debug_print(func.__name__, "type", x=type(x), theta=type(theta))
            return None
        m, n = x.shape
        if theta.shape != (n + 1, 1):
            _debug_print(func.__name__, "shape", x=x.shape, theta=theta.shape)
            return None
        return func(x, theta, *args, **kwargs)

    return wrapper


def check_type_and_shape_vector_pair(func):
    @functools.wraps(func)
    def wrapper(y, y_hat, *args, **kwargs):
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            _debug_print(func.__name__, "type", y=type(y), y_hat=type(y_hat))
            return None
        if y.shape != y_hat.shape or y.shape[1] != 1:
            _debug_print(func.__name__, "shape", y=y.shape, y_hat=y_hat.shape)
            return None
        return func(y, y_hat, *args, **kwargs)

    return wrapper


def check_type_and_shape_vector(func):
    @functools.wraps(func)
    def wrapper(x, *args, **kwargs):
        if not isinstance(x, np.ndarray):
            _debug_print(func.__name__, "type", x=type(x))
            return None
        if x.shape[1] != 1:
            _debug_print(func.__name__, "shape", x=x.shape)
            return None
        return func(x, *args, **kwargs)

    return wrapper


def check_type_and_shape_vector_any(func):
    @functools.wraps(func)
    def wrapper(x, *args, **kwargs):
        if not isinstance(x, np.ndarray):
            _debug_print(func.__name__, "type", x=type(x))
            return None
        if x.size == 0:
            _debug_print(func.__name__, "empty", x=x.size)
            return None
        return func(x, *args, **kwargs)

    return wrapper


def check_type_and_shape_vector_pair_any(func):
    @functools.wraps(func)
    def wrapper(y, y_hat, *args, **kwargs):
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            _debug_print(func.__name__, "type", y=type(y), y_hat=type(y_hat))
            return None
        if y.size == 0 or y_hat.size == 0:
            _debug_print(func.__name__, "empty", y=y.size, y_hat=y_hat.size)
            return None
        if y.shape != y_hat.shape:
            _debug_print(func.__name__, "shape", y=y.shape, y_hat=y_hat.shape)
            return None
        return func(y, y_hat, *args, **kwargs)

    return wrapper


def check_type_and_shape_fit(func):
    @functools.wraps(func)
    def wrapper(x, y, theta, alpha, *args, **kwargs):
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(y, np.ndarray)
            or not isinstance(theta, np.ndarray)
            or not isinstance(alpha, float)
        ):
            _debug_print(
                func.__name__,
                "type",
                x=type(x),
                y=type(y),
                theta=type(theta),
                alpha=type(alpha),
            )
            return None
        m, n = x.shape
        if y.shape != (m, 1) or theta.shape != (n + 1, 1):
            _debug_print(
                func.__name__, "shape", x=x.shape, y=y.shape, theta=theta.shape
            )
            return None
        return func(x, y, theta, alpha, *args, **kwargs)

    return wrapper


def check_type_and_shape_theta(func):
    @functools.wraps(func)
    def wrapper(theta, *args, **kwargs):
        if not isinstance(theta, np.ndarray):
            _debug_print(func.__name__, "type", theta=type(theta))
            return None
        if theta.shape[1] != 1:
            _debug_print(func.__name__, "shape", theta=theta.shape)
            return None
        return func(theta, *args, **kwargs)

    return wrapper


def check_type_and_shape_reg_loss(func):
    @functools.wraps(func)
    def wrapper(y, y_hat, theta, lambda_, *args, **kwargs):
        if (
            not isinstance(y, np.ndarray)
            or not isinstance(y_hat, np.ndarray)
            or not isinstance(theta, np.ndarray)
            or not isinstance(lambda_, float)
        ):
            _debug_print(
                func.__name__,
                "type",
                y=type(y),
                y_hat=type(y_hat),
                theta=type(theta),
                lambda_=type(lambda_),
            )
            return None
        if y.shape != y_hat.shape or y.shape[1] != 1 or theta.shape[1] != 1:
            _debug_print(
                func.__name__, "shape", y=y.shape, y_hat=y_hat.shape, theta=theta.shape
            )
            return None
        return func(y, y_hat, theta, lambda_, *args, **kwargs)

    return wrapper


def check_type_and_shape_polynomial(func):
    @functools.wraps(func)
    def wrapper(x, power, *args, **kwargs):
        if not isinstance(x, np.ndarray) or not isinstance(power, int):
            _debug_print(func.__name__, "type", x=type(x), power=type(power))
            return None
        return func(x, power, *args, **kwargs)

    return wrapper


def check_type_and_shape_confusion(func):
    @functools.wraps(func)
    def wrapper(y_true, y_hat, *args, **kwargs):
        if not isinstance(y_true, np.ndarray) or not isinstance(y_hat, np.ndarray):
            _debug_print(func.__name__, "type", y_true=type(y_true), y_hat=type(y_hat))
            return None
        if y_true.shape != y_hat.shape:
            _debug_print(func.__name__, "shape", y_true=y_true.shape, y_hat=y_hat.shape)
            return None
        return func(y_true, y_hat, *args, **kwargs)

    return wrapper


def check_type_and_shape_plot(func):
    @functools.wraps(func)
    def wrapper(x, y, theta, *args, **kwargs):
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(y, np.ndarray)
            or not isinstance(theta, np.ndarray)
        ):
            _debug_print(func.__name__, "type", x=type(x), y=type(y), theta=type(theta))
            return None
        if x.shape[0] != y.shape[0] or x.ndim != 1 or y.ndim != 1 or theta.ndim != 2:
            _debug_print(
                func.__name__, "shape", x=x.shape, y=y.shape, theta=theta.shape
            )
            return None
        return func(x, y, theta, *args, **kwargs)

    return wrapper


def check_type_and_shape_split(func):
    @functools.wraps(func)
    def wrapper(x, y, proportion, *args, **kwargs):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            _debug_print(func.__name__, "type", x=type(x), y=type(y))
            return None
        if x.size == 0 or y.size == 0:
            _debug_print(func.__name__, "empty", x=x.size, y=y.size)
            return None
        if y.shape[0] != x.shape[0]:
            _debug_print(func.__name__, "shape", x=x.shape, y=y.shape)
            return None
        if not isinstance(proportion, (float, int)):
            _debug_print(func.__name__, "type", proportion=type(proportion))
            return None
        return func(x, y, proportion, *args, **kwargs)

    return wrapper


def check_type_and_shape_fit_method(func):
    @functools.wraps(func)
    def wrapper(self, x, y, *args, **kwargs):
        theta = getattr(self, "thetas", None)
        alpha = getattr(self, "alpha", None)
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(y, np.ndarray)
            or not isinstance(theta, np.ndarray)
            or not isinstance(alpha, (float, int))
        ):
            _debug_print(
                func.__name__,
                "type",
                x=type(x),
                y=type(y),
                theta=type(theta),
                alpha=type(alpha),
            )
            return None
        m, n = x.shape
        if y.shape != (m, 1) or theta.shape != (n + 1, 1):
            _debug_print(
                func.__name__, "shape", x=x.shape, y=y.shape, theta=theta.shape
            )
            return None
        return func(self, x, y, *args, **kwargs)

    return wrapper
