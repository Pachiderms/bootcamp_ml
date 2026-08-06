import numpy as np
import pandas as pd
from src.data_spliter import data_spliter
from src.standardization import zscore
from src.linear_regression import MyLinearRegression as MyLR
from src.ridge import MyRidge
from src.linear_regression import add_polynomial_features_mult
from src.polynomial_model_extended import add_polynomial_features
from src.benchmark_train import train_models_from_csv, train_models_reg_from_csv
import matplotlib.pyplot as plt
import ast


def split_set(file="../attachments/space_avocado.csv", ratio=0.75):
    df = pd.read_csv(file)

    X = zscore(np.array(df[["weight", "prod_distance", "time_delivery"]]))
    Y = zscore(np.array(df[["target"]]))

    x_train, x_test, y_train, y_test = data_spliter(X, Y, ratio)
    return (x_train, x_test, y_train, y_test)


def final_train():
    x_train, x_test, y_train, y_test = split_set()
    xtr, xtst = add_polynomial_features_mult(x_train, 2), add_polynomial_features_mult(
        x_test, 2
    )
    best_model = MyLR(np.ones(xtr.shape[1] + 1), alpha=5e-5, max_iter=50000).fit_(
        xtr, y_train
    )
    y_hat = best_model.predict_(xtst)

    models, errors, predictions = train_models_from_csv(
        x_train, x_test, y_train, y_test, [1, 3, 4]
    )

    models.insert(1, best_model)
    errors.insert(1, best_model.mse_(y_test, y_hat))
    predictions.insert(1, y_hat)

    plt.plot([1, 2, 3, 4], errors, marker="o", color="yellowgreen")
    plt.xlabel("Degree")
    plt.ylabel("mse")

    weight = x_test[:, 0]
    prod_distance = x_test[:, 1]
    time_delivery = x_test[:, 2]
    y_hat = predictions[0]

    plt.figure(figsize=(18, 6))

    plt.scatter(weight, time_delivery, c=y_test)
    plt.scatter(weight, time_delivery, c=predictions[1], marker="x", s=40, alpha=0.4)

    plt.scatter(weight, prod_distance, c=y_test)
    plt.scatter(weight, prod_distance, c=predictions[1], marker="x", s=40, alpha=0.4)

    plt.scatter(time_delivery, prod_distance, c=y_test)
    plt.scatter(
        time_delivery, prod_distance, c=predictions[1], marker="x", s=40, alpha=0.4
    )


def final_train_reg(errors):
    _, x_test, _, y_test = split_set()

    best_model = np.argmin(errors)
    df = pd.read_csv("../attachments/models_reg.csv")
    degree = df["degree"].iloc[best_model]
    df = df[df["degree"] == degree]

    xtst = add_polynomial_features(x_test, int(degree))

    for i in range(len(df)):
        thetas = df["thetas"].apply(lambda x: np.array(ast.literal_eval(x))).iloc[i]
        alpha = df["alpha"].iloc[i]
        max_iter = df["max_iter"].iloc[i]
        lambda_ = df["lambdas"].iloc[i]
        model = MyRidge(thetas=[0])
        model.set_params(
            **{
                "thetas": thetas,
                "alpha": float(alpha),
                "max_iter": int(max_iter),
                "lambda_": lambda_,
            }
        )
        y_hat = model.predict_(xtst)

        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        ax[0].scatter(y_test, y_hat, alpha=0.5)
        ax[0].plot([y_hat.min(), y_hat.max()], [y_hat.min(), y_hat.max()], "r--")
        ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "y--")

        ax[0].set_xlabel("True values")
        ax[0].set_ylabel("Predicted values")
        ax[0].set_title(f"Predicted vs True for lamda_={float(lambda_):.1f}")

        ax[1].plot(y_test, label="True")
        ax[1].plot(y_hat, label="Predicted")
        ax[1].legend()
        ax[1].set_title(f"Predicted vs True for lamda_={float(lambda_):.1f}")

    plt.show()
