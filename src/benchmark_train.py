import numpy as np
from src.linear_regression import add_polynomial_features_mult
from src.linear_regression import MyLinearRegression as MyLR
from src.ridge import MyRidge
from src.polynomial_model_extended import add_polynomial_features
import pandas as pd
import ast


def train_models(x_train, x_test, y_train, y_test, degree):
    models = []
    errors = []
    predictions = []

    for d in range(1, degree + 1):
        if d == 1:
            xtr, xtst = x_train, x_test
        else:
            xtr, xtst = add_polynomial_features_mult(
                x_train, d
            ), add_polynomial_features_mult(x_test, d)

        m = MyLR(thetas=np.ones(xtr.shape[1] + 1), alpha=5e-5, max_iter=50000).fit_(
            xtr, y_train
        )
        models.append(m)
        y_hat = m.predict_(xtst)
        predictions.append(y_hat)
        errors.append(m.mse_(y_test, y_hat))

    df = pd.DataFrame(
        data={
            "degree": [i for i in range(1, degree + 1)],
            "thetas": [m.thetas.tolist() for m in models],
            "alpha": [m.alpha for m in models],
            "max_iter": [m.max_iter for m in models],
        }
    )

    df.to_csv(path_or_buf="../attachments/models.csv", index=False)
    return (models, errors, predictions)


def train_models_from_csv(
    x_train, x_test, y_train, y_test, degrees, file="../attachments/models.csv"
):
    df = pd.read_csv(file)
    df["thetas"] = df["thetas"].apply(lambda x: np.array(ast.literal_eval(x)))
    df = df[df["degree"].isin(degrees)]
    deg = df["degree"]
    thetas = df["thetas"]
    alphas = df["alpha"]
    max_iters = df["max_iter"]
    models = []
    errors = []
    predictions = []

    for d in range(1, len(degrees) + 1):
        degree = int(deg.iloc[d - 1])
        if degree == 1:
            xtr, xtst = x_train, x_test
        else:
            xtr, xtst = add_polynomial_features_mult(
                x_train, degree
            ), add_polynomial_features_mult(x_test, degree)

        m = MyLR(
            thetas=thetas.iloc[d - 1],
            alpha=alphas.iloc[d - 1],
            max_iter=max_iters.iloc[d - 1],
        ).fit_(xtr, y_train)
        models.append(m)
        y_hat = m.predict_(xtst)
        predictions.append(y_hat)
        errors.append(m.mse_(y_test, y_hat))

    return (models, errors, predictions)


def train_models_reg(x_train, x_test, y_train, y_test, degree):
    models = []
    degrees = []
    errors = []
    predictions = []

    for d in range(1, degree + 1):
        if d == 1:
            xtr, xtst = x_train, x_test
        else:
            xtr, xtst = add_polynomial_features_mult(
                x_train, d
            ), add_polynomial_features_mult(x_test, d)

        lambdas_ = np.linspace(0, 1.0, num=6)
        for lambda_ in lambdas_:
            m = MyRidge(
                thetas=np.ones(xtr.shape[1] + 1),
                alpha=5e-5,
                max_iter=50000,
                lambda_=lambda_,
            ).fit_(xtr, y_train)
            models.append(m)
            degrees.append(d)
            y_hat = m.predict_(xtst)
            predictions.append(y_hat)
            errors.append(m.mse_(y_test, y_hat))

    df = pd.DataFrame(
        data={
            "degree": [d for d in degrees],
            "thetas": [m.thetas.tolist() for m in models],
            "alpha": [m.alpha for m in models],
            "max_iter": [m.max_iter for m in models],
            "lambdas": [m.lambda_ for m in models],
        }
    )

    df.to_csv(path_or_buf="../attachments/models_reg.csv", index=False)
    return (models, errors, predictions)


def train_models_reg_from_csv(
    x_train, x_test, y_train, y_test, degrees, file="../attachments/models_reg.csv"
):
    df = pd.read_csv(file)
    df["thetas"] = df["thetas"].apply(lambda x: np.array(ast.literal_eval(x)))
    df = df[df["degree"].isin(degrees)]
    deg = df["degree"]
    thetas = df["thetas"]
    alphas = df["alpha"]
    max_iters = df["max_iter"]
    lambdas_ = df["lambdas"]
    models = []
    errors = []
    predictions = []

    for d in range(1, len(degrees) + 1):
        degree = int(deg.iloc[d - 1])
        if degree == 1:
            xtr, xtst = x_train, x_test
        else:
            xtr, xtst = add_polynomial_features(
                x_train, degree
            ), add_polynomial_features(x_test, degree)

        m = MyRidge(
            thetas=thetas.iloc[d - 1],
            alpha=alphas.iloc[d - 1],
            max_iter=max_iters.iloc[d - 1],
            lambda_=lambdas_.iloc[d - 1],
        ).fit_(xtr, y_train)
        models.append(m)
        y_hat = m.predict_(xtst)
        predictions.append(y_hat)
        errors.append(m.mse_(y_test, y_hat))

    return (models, errors, predictions)
