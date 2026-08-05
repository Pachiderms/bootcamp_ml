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


def final_train_reg():
    x_train, x_test, y_train, y_test = split_set()
    xtr, xtst = add_polynomial_features(x_train, 2), add_polynomial_features(x_test, 2)
    best_model = MyRidge(
        np.ones(xtr.shape[1] + 1), alpha=5e-5, max_iter=50000, lambda_=0.2
    ).fit_(xtr, y_train)
    y_hat = best_model.predict_(xtst)

    models, errors, predictions = train_models_reg_from_csv(
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
