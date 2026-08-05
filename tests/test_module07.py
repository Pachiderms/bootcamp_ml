import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from src.linear_regression import add_polynomial_features, MyLinearRegression as MyLR
from src.plot import MyPloter as MyPlt
from src.fit import fit_
from src.gradient import gradient
from src.loss import loss_
from src.prediction import simple_predict, predict_
from src.standardization import zscore
from src.space_avocado import split_set
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from src.benchmark_train import train_models
from src.linear_regression import add_polynomial_features_mult


def test_func():
    x = np.arange(1, 13).reshape((4, -1))

    theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
    assert np.allclose(
        simple_predict(x, theta1), np.array([[5.0], [5.0], [5.0], [5.0]])
    )

    theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
    assert np.allclose(
        simple_predict(x, theta2), np.array([[1.0], [4.0], [7.0], [10.0]])
    )

    theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
    assert np.allclose(
        simple_predict(x, theta3), np.array([[9.64], [24.28], [38.92], [53.56]])
    )

    theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
    assert np.allclose(
        simple_predict(x, theta4), np.array([[12.5], [32.0], [51.5], [71.0]])
    )

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


def test_my_lr():
    X = np.array(
        [[1.0, 1.0, 2.0, 3.0], [5.0, 8.0, 13.0, 21.0], [34.0, 55.0, 89.0, 144.0]]
    )
    Y = np.array([[23.0], [48.0], [218.0]])
    mylr = MyLR([[1.0], [1.0], [1.0], [1.0], [1]])

    y_hat = mylr.predict_(X)
    assert np.allclose(y_hat, np.array([[8.0], [48.0], [323.0]]))
    assert np.allclose(mylr.loss_elem_(Y, y_hat), np.array([[225.0], [0.0], [11025.0]]))
    assert np.allclose(mylr.loss_(Y, y_hat), 1875.0)

    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.thetas = np.ones((X.shape[1] + 1, 1))
    mylr.fit_(X, Y)

    assert np.allclose(
        mylr.thetas,
        np.array([[18.188], [2.767], [-0.374], [1.392], [0.017]]),
        rtol=1e-1,
    )

    y_hat = mylr.predict_(X)
    assert np.allclose(y_hat, np.array([[23.417], [47.489], [218.065]]), rtol=1e-2)
    assert np.allclose(
        mylr.loss_elem_(Y, y_hat), np.array([[0.174], [0.260], [0.004]]), rtol=1e-1
    )
    assert np.allclose(mylr.loss_(Y, y_hat), 0.0732, rtol=1e-2)

    df = pd.read_csv("../attachments/spacecraft_data.csv")
    Age = np.array(df[["Age"]])
    Sell_price = np.array(df[["Sell_price"]])
    Thrust_power = np.array(df[["Thrust_power"]])
    Terameters = np.array(df[["Terameters"]])

    figures = 3
    plotter = MyPlt((1, figures))
    plotter.scatter(
        Age,
        Sell_price,
        xlabel=f"$x_{{\\text{{1}}}}$: age (in years)",
        ylabel="y: sell price (in keuros)",
        label="Sell price",
    )
    plotter.scatter(
        Thrust_power,
        Sell_price,
        xlabel=f"$x_{{\\text{{2}}}}$: thrust power(in 10Km/s)",
        ylabel="y: sell price (in keuros)",
        label="Sell price",
        color="green",
        ax_id=1,
    )
    plotter.scatter(
        Terameters,
        Sell_price,
        xlabel=f"$x_{{\\text{{3}}}}$: distance totalizer value of spacecraft (in Tmeters)",
        ylabel="y: sell price (in keuros)",
        label="Sell price",
        color="darkviolet",
        ax_id=2,
    )

    myLR_age = MyLR([[1000.0], [-1.0]], alpha=5e-5, max_iter=500000)
    myLR_age.fit_(Age, Sell_price)
    y_hat = myLR_age.predict_(Age)

    myLR_thrust = MyLR([[1.0], [-1.0]], alpha=5e-5, max_iter=500000)
    myLR_thrust.fit_(Thrust_power, Sell_price)
    y_hat2 = myLR_thrust.predict_(Thrust_power)

    myLR_tm = MyLR([[1.0], [-1.0]], alpha=5e-5, max_iter=500000)
    myLR_tm.fit_(Terameters, Sell_price)
    y_hat3 = myLR_tm.predict_(Terameters)

    plotter.plot(
        Age,
        y_hat,
        param_dict={
            "color": "cyan",
            "linestyle": "dotted",
            "label": "Predicted sell price",
        },
    )

    plotter.plot(
        Thrust_power,
        y_hat2,
        param_dict={
            "color": "lime",
            "linestyle": "dotted",
            "label": "Predicted sell price",
        },
        ax_id=1,
    )
    plotter.plot(
        Terameters,
        y_hat3,
        param_dict={
            "color": "violet",
            "linestyle": "dotted",
            "label": "Predicted sell price",
        },
        ax_id=2,
    )
    print(myLR_age.mse_(Sell_price, y_hat))
    print(myLR_thrust.mse_(Sell_price, y_hat2))
    print(myLR_tm.mse_(Sell_price, y_hat3))

    X = np.array(df[["Age", "Thrust_power", "Terameters"]])
    Y = np.array(df[["Sell_price"]])

    my_lreg = MyLR(thetas=[1.0, 1.0, 1.0, 1.0], alpha=9e-5, max_iter=500000)

    y_hat = my_lreg.predict_(X)

    assert np.allclose(my_lreg.mse_(Y, y_hat), 144044.877, rtol=1e-2)
    my_lreg.fit_(X, Y)
    my_lreg.plot_learning_curve()

    assert np.allclose(
        my_lreg.thetas, np.array([[367.288], [-23.699], [5.736], [-2.638]]), rtol=1e-2
    )

    y_hat2 = my_lreg.predict_(X)
    assert np.allclose(my_lreg.mse_(Y, y_hat2), 435.9325695, rtol=1e-2)

    figures = 3
    plotter = MyPlt((1, figures), (15, 5))
    plotter.scatter(
        Age,
        Y,
        xlabel=f"$x_{{\\text{{1}}}}$: age (in years)",
        ylabel="y: sell price (in keuros)",
        label="Sell price",
    )
    plotter.scatter(
        Thrust_power,
        Y,
        xlabel=f"$x_{{\\text{{2}}}}$: thrust power(in 10Km/s)",
        ylabel="y: sell price (in keuros)",
        label="Sell price",
        color="green",
        ax_id=1,
    )
    plotter.scatter(
        Terameters,
        Y,
        xlabel=f"$x_{{\\text{{3}}}}$: distance totalizer value of spacecraft (in Tmeters)",
        ylabel="y: sell price (in keuros)",
        label="Sell price",
        color="darkviolet",
        ax_id=2,
    )

    plotter.scatter(
        Age,
        y_hat2,
        xlabel=f"$x_{{\\text{{1}}}}$: age (in years)",
        ylabel="y: sell price (in keuros)",
        label="Sell price",
        color="deepskyblue",
        s=5,
    )
    plotter.scatter(
        Thrust_power,
        y_hat2,
        xlabel=f"$x_{{\\text{{2}}}}$: thrust power(in 10Km/s)",
        ylabel="y: sell price (in keuros)",
        label="Sell price",
        color="lime",
        s=5,
        ax_id=1,
    )
    plotter.scatter(
        Terameters,
        y_hat2,
        xlabel=f"$x_{{\\text{{3}}}}$: distance totalizer value of spacecraft (in Tmeters)",
        ylabel="y: sell price (in keuros)",
        label="Sell price",
        color="violet",
        s=5,
        ax_id=2,
    )


def test_add_polynomial_features():
    x = np.arange(1, 6).reshape(-1, 1)

    print(add_polynomial_features(x, 3))
    print(
        sklearn.preprocessing.PolynomialFeatures(degree=3, include_bias=False)
        .fit_transform(x)
        .astype(int)
    )
    assert np.array_equal(
        add_polynomial_features(x, 6),
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [2, 4, 8, 16, 32, 64],
                [3, 9, 27, 81, 243, 729],
                [4, 16, 64, 256, 1024, 4096],
                [5, 25, 125, 625, 3125, 15625],
            ]
        ),
    )
    assert np.array_equal(
        sklearn.preprocessing.PolynomialFeatures(degree=6, include_bias=False)
        .fit_transform(x)
        .astype(int),
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [2, 4, 8, 16, 32, 64],
                [3, 9, 27, 81, 243, 729],
                [4, 16, 64, 256, 1024, 4096],
                [5, 25, 125, 625, 3125, 15625],
            ]
        ),
    )

    print(add_polynomial_features(x, 6))
    print(
        sklearn.preprocessing.PolynomialFeatures(degree=6, include_bias=False)
        .fit_transform(x)
        .astype(int)
    )
    assert np.array_equal(
        add_polynomial_features(x, 6),
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [2, 4, 8, 16, 32, 64],
                [3, 9, 27, 81, 243, 729],
                [4, 16, 64, 256, 1024, 4096],
                [5, 25, 125, 625, 3125, 15625],
            ]
        ),
    )
    assert np.array_equal(
        sklearn.preprocessing.PolynomialFeatures(degree=6, include_bias=False)
        .fit_transform(x)
        .astype(int),
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [2, 4, 8, 16, 32, 64],
                [3, 9, 27, 81, 243, 729],
                [4, 16, 64, 256, 1024, 4096],
                [5, 25, 125, 625, 3125, 15625],
            ]
        ),
    )

    x = np.arange(1, 11).reshape(-1, 1)
    y = np.array(
        [
            [1.39270298],
            [3.88237651],
            [4.37726357],
            [4.63389049],
            [7.79814439],
            [6.41717461],
            [8.63429886],
            [8.19939795],
            [10.37567392],
            [10.68238222],
        ]
    )

    x_ = add_polynomial_features(x, 3)
    x_norm = zscore(x_)
    my_lr = MyLR(np.ones(4).reshape(-1, 1), alpha=1e-1, max_iter=15000).fit_(x_norm, y)
    my_lr.plot_learning_curve()
    # Plot:
    ## To get a smooth curve, we need a lot of data points
    continuous_x = np.arange(1, 10.01, 0.01).reshape(-1, 1)
    x_plot = add_polynomial_features(continuous_x, 3)
    x_plot_norm = zscore(x_plot)
    y_hat = my_lr.predict_(x_plot_norm)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].scatter(x, y)
    axs[1].scatter(x, y)
    axs[1].plot(continuous_x, y_hat, color="orange")


def test_my_lr_polynomial():
    def fit_zscore(x):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std[std == 0] = 1
        return (x - mean) / std, mean, std

    def transform_zscore(x, mean, std):
        return (x - mean) / std

    df = pd.read_csv("../attachments/are_blue_pills_magics.csv")
    x = np.array(df[["Micrograms"]])
    y = np.array(df[["Score"]])

    theta1 = np.array([[1.0, 1.0]])
    theta2 = np.array([[1.0, 1.0, 1.0]])
    theta3 = np.array([[1.0, 1.0, 1.0, 1.0]])
    theta4 = np.array([[-20], [160], [-80], [10], [-1]]).reshape(-1, 1)
    theta5 = np.array([[1140], [-1850], [1110], [-305], [40], [-2]]).reshape(-1, 1)
    theta6 = np.array(
        [[9110], [-18015], [13400], [-4935], [966], [-96.4], [3.86]]
    ).reshape(-1, 1)

    x2 = add_polynomial_features(x, 2)
    x3 = add_polynomial_features(x, 3)
    x4 = add_polynomial_features(x, 4)
    x5 = add_polynomial_features(x, 5)
    x6 = add_polynomial_features(x, 6)

    x_norm, mean1, std1 = fit_zscore(x)
    x2_norm, mean2, std2 = fit_zscore(x2)
    x3_norm, mean3, std3 = fit_zscore(x3)
    x4_norm, mean4, std4 = fit_zscore(x4)
    x5_norm, mean5, std5 = fit_zscore(x5)
    x6_norm, mean6, std6 = fit_zscore(x6)

    lr1 = MyLR(theta1, alpha=5.5e-4, max_iter=50000).fit_(x_norm, y)
    lr2 = MyLR(theta2, alpha=5.5e-4, max_iter=50000).fit_(x2_norm, y)
    lr3 = MyLR(theta3, alpha=5.5e-4, max_iter=50000).fit_(x3_norm, y)
    lr4 = MyLR(theta4, alpha=5.5e-4, max_iter=50000).fit_(x4_norm, y)
    lr5 = MyLR(theta5, alpha=5.5e-4, max_iter=50000).fit_(x5_norm, y)
    lr6 = MyLR(theta6, alpha=5.5e-4, max_iter=50000).fit_(x6_norm, y)

    y_hat = lr1.predict_(x_norm)
    y_hat2 = lr2.predict_(x2_norm)
    y_hat3 = lr3.predict_(x3_norm)
    y_hat4 = lr4.predict_(x4_norm)
    y_hat5 = lr5.predict_(x5_norm)
    y_hat6 = lr6.predict_(x6_norm)

    mse = lr1.mse_(y, y_hat)
    mse2 = lr2.mse_(y, y_hat2)
    mse3 = lr3.mse_(y, y_hat3)
    mse4 = lr4.mse_(y, y_hat4)
    mse5 = lr5.mse_(y, y_hat5)
    mse6 = lr6.mse_(y, y_hat6)

    print(f"{mse=}")
    print(f"{mse2=}")
    print(f"{mse3=}")
    print(f"{mse4=}")
    print(f"{mse5=}")
    print(f"{mse6=}")

    poly_degrees = np.arange(1, 7, 1)
    mse_scores = np.array([mse, mse2, mse3, mse4, mse5, mse6])
    mse_scores_norm = zscore(mse_scores)
    bar_labels = ["deg=1", "deg=2", "deg=3", "deg=4", "deg=5", "deg=6"]
    bar_colors = ["lime", "green", "limegreen", "orange", "orangered", "red"]
    fig, ax = plt.subplots()
    ax.bar(poly_degrees, mse_scores, label=bar_labels, color=bar_colors)
    ax.set_yscale("log")
    ax.set_ylabel("mse")

    plt.legend()
    plt.show()

    lr1.plot_learning_curve()
    lr2.plot_learning_curve()
    lr3.plot_learning_curve()
    lr4.plot_learning_curve()
    lr5.plot_learning_curve()
    lr6.plot_learning_curve()

    continuous_x = np.arange(1, 7.01, 0.01).reshape(-1, 1)
    x_plot = continuous_x
    x_plot_2 = add_polynomial_features(continuous_x, 2)
    x_plot_3 = add_polynomial_features(continuous_x, 3)
    x_plot_4 = add_polynomial_features(continuous_x, 4)
    x_plot_5 = add_polynomial_features(continuous_x, 5)
    x_plot_6 = add_polynomial_features(continuous_x, 6)

    x_plotnorm = transform_zscore(x_plot, mean1, std1)
    x2_plotnorm = transform_zscore(x_plot_2, mean2, std2)
    x3_plotnorm = transform_zscore(x_plot_3, mean3, std3)
    x4_plotnorm = transform_zscore(x_plot_4, mean4, std4)
    x5_plotnorm = transform_zscore(x_plot_5, mean5, std5)
    x6_plotnorm = transform_zscore(x_plot_6, mean6, std6)

    y_hat = lr1.predict_(x_plotnorm)
    y_hat2 = lr2.predict_(x2_plotnorm)
    y_hat3 = lr3.predict_(x3_plotnorm)
    y_hat4 = lr4.predict_(x4_plotnorm)
    y_hat5 = lr5.predict_(x5_plotnorm)
    y_hat6 = lr6.predict_(x6_plotnorm)

    plotter = MyPlt((1, 6), (20, 10))
    plotter.scatter(x, y, xlabel="Micrograms", ylabel="Score")
    plotter.scatter(x, y, xlabel="Micrograms", ylabel="Score", ax_id=1)
    plotter.scatter(x, y, xlabel="Micrograms", ylabel="Score", ax_id=2)
    plotter.scatter(x, y, xlabel="Micrograms", ylabel="Score", ax_id=3)
    plotter.scatter(x, y, xlabel="Micrograms", ylabel="Score", ax_id=4)
    plotter.scatter(x, y, xlabel="Micrograms", ylabel="Score", ax_id=5)

    plotter.plot(
        continuous_x,
        y_hat,
        param_dict={
            "color": "cyan",
            "linestyle": "dotted",
            "label": "Predicted sell price",
        },
    )
    plotter.plot(
        continuous_x,
        y_hat2,
        param_dict={
            "color": "coral",
            "linestyle": "dotted",
            "label": "Predicted sell price",
        },
        ax_id=1,
    )
    plotter.plot(
        continuous_x,
        y_hat3,
        param_dict={
            "color": "dodgerblue",
            "linestyle": "dotted",
            "label": "Predicted sell price",
        },
        ax_id=2,
    )
    plotter.plot(
        continuous_x,
        y_hat4,
        param_dict={
            "color": "navy",
            "linestyle": "dotted",
            "label": "Predicted sell price",
        },
        ax_id=3,
    )
    plotter.plot(
        continuous_x,
        y_hat5,
        param_dict={
            "color": "green",
            "linestyle": "dotted",
            "label": "Predicted sell price",
        },
        ax_id=4,
    )

    plotter.plot(
        continuous_x,
        y_hat6,
        param_dict={
            "color": "violet",
            "linestyle": "dotted",
            "label": "Predicted sell price",
        },
        ax_id=5,
    )


def test_train_vs_test():
    x_train, x_test, y_train, y_test = split_set()

    models, errors, predictions = train_models(x_train, x_test, y_train, y_test, 4)

    print(f"{errors=}")

    plt.plot([1, 2, 3, 4], errors, marker="o", color="yellowgreen")
    plt.xlabel("Degree")
    plt.ylabel("mse")

    plt.show()

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

    min_err = np.argmin(errors)
    print(min_err)
    best_model = models[min_err]

    xtst = add_polynomial_features_mult(x_test, 2)
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
