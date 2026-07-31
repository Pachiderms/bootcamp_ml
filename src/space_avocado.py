import numpy as np
import pandas as pd
from src.data_spliter import data_spliter
from src.standardization import zscore
from src.linear_regression import MyLinearRegression as MyLR
from src.linear_regression import add_polynomial_features_mult
from src.benchmark_train import train_models_from_csv
import matplotlib.pyplot as plt
from src.linear_regression import MyPloter as MyPlt

def split_set(file="../attachments/space_avocado.csv", ratio=0.75):
    df = pd.read_csv(file)

    X = zscore(np.array(df[['weight', 'prod_distance', 'time_delivery']]))
    Y = zscore(np.array(df[["target"]]))

    (x_train, x_test, y_train, y_test) = data_spliter(X, Y, ratio)
    return (x_train, x_test, y_train, y_test)


def final_train():
    (x_train, x_test, y_train, y_test) = split_set()
    xtr, xtst = add_polynomial_features_mult(x_train, 2), add_polynomial_features_mult(x_test, 2)
    best_model = MyLR(np.ones(xtr.shape[1] + 1), alpha=5e-5, max_iter=50000).fit_(xtr, y_train)
    y_hat = best_model.predict_(xtst)

    models, errors, predictions = train_models_from_csv(x_train, x_test, y_train, y_test, [1, 3, 4])

    models.insert(1, best_model)
    errors.insert(1, best_model.mse_(y_test, y_hat))
    predictions.insert(1, y_hat)

    plt.plot([1, 2, 3, 4], errors, marker='o', color='yellowgreen')
    plt.xlabel("Degree")
    plt.ylabel("mse")

    plt.show()

    plotter = MyPlt((1, 3), (15, 5))
    plotter.scatter(x_test[:, [0]], y_test, xlabel="weight", ylabel="sell price", label="target", s=10)
    plotter.scatter(x_test[:, [1]], y_test, xlabel="prod_distance", ylabel="sell price", label="target", color="green", ax_id=1, s=10)
    plotter.scatter(x_test[:, [2]], y_test, xlabel="time_delivery", ylabel="sell price", label="target", color="darkviolet", ax_id=2, s=10)

    plotter.scatter(x_test[:, [0]], predictions[1], color="deepskyblue", s=5, label="Sell price")
    plotter.scatter(x_test[:, [1]], predictions[1], color="lime", s=5, ax_id=1, label="Sell price")
    plotter.scatter(x_test[:, [2]], predictions[1], color="violet", s=5, ax_id=2, label="Sell price")
