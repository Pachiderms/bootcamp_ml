from src.data_spliter import data_spliter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.polynomial_model_extended import add_polynomial_features
from src.other_metrics import f1_score_
from src.logistic_regression import MyLogisticRegression as MyLR
from src.logistic_regression import MyLogisticRegression as MyLogR


def one_vs_all():
    """
    Description:
        This program does:
        1. Split the dataset into a training and a test set.
        2. Train 4 logistic regression classifiers to discriminate each class from the others (the
        way you did in part one).
        3. Predict for each example the class according to each classifier and select the one
        with the highest output probability score.
        4. Calculate and display the fraction of correct predictions over the total number of
        predictions based on the test set.
        5. Plot 3 scatter plots (one for each pair of citizen features) with the dataset and the
        final prediction of the model.
    Returns:
        Nothing.
    """
    df1 = pd.read_csv("../attachments/solar_system_census.csv")
    df2 = pd.read_csv("../attachments/solar_system_census_planets.csv")

    df = pd.concat([df1, df2["Origin"]], axis=1)
    X = np.array(df.iloc[:, 1:4])
    Y = np.array(df["Origin"]).reshape(-1, 1)
    x_train, x_test, y_train_full, y_test_full = data_spliter(X, Y, 0.7)
    y_train = (y_train_full == planet).astype(int)

    train_min = x_train.min(axis=0)
    train_max = x_train.max(axis=0)

    x_train_norm = (x_train - train_min) / (train_max - train_min)
    x_test_norm = (x_test - train_min) / (train_max - train_min)
    models = []
    probas = []

    for planet in range(4):
        model = MyLR(
            thetas=np.ones((x_train.shape[1] + 1, 1)), alpha=1e-1, max_iter=50000
        ).fit_(x_train_norm, y_train)
        models.append(model)

        proba = model.predict_(x_test_norm)
        probas.append(proba)

    scores = np.column_stack([proba for proba in probas])
    prediction = np.argmax(scores, axis=1).reshape(-1, 1)

    match = prediction == y_test_full
    print(f"eval predictions: {np.sum(match) * 100 / len(y_test_full)}% accurate.\n\
({np.sum(match)} good predictions out of {len(y_test_full)}).")

    height = x_test[:, 0]
    weight = x_test[:, 1]
    bone_density = x_test[:, 2]

    _, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].scatter(weight, height, c=y_test_full.ravel(), alpha=0.4)
    axs[0].scatter(weight, height, c=prediction.ravel(), marker="x")
    axs[0].set_xlabel("Weight")
    axs[0].set_ylabel("Height")
    axs[0].set_title("Weight vs Height")

    axs[1].scatter(weight, bone_density, c=y_test_full.ravel(), alpha=0.4)
    axs[1].scatter(weight, bone_density, c=prediction.ravel(), marker="x")
    axs[1].set_xlabel("Weight")
    axs[1].set_ylabel("Bone density")
    axs[1].set_title("Weight vs Bone Density")

    axs[2].scatter(height, bone_density, c=y_test_full.ravel(), alpha=0.4)
    axs[2].scatter(height, bone_density, c=prediction.ravel(), marker="x")
    axs[2].set_xlabel("Height")
    axs[2].set_ylabel("Bone density")
    axs[2].set_title("Height vs Bone Density")

    plt.show()


def one_vs_all_log():
    """
    Description:
        This program does:
        1. Split the dataset into a training and a test set.
        2. Train 24 logistic regression classifiers to discriminate each class from the others (the
        way you did in part one).
        3. Predict for each example the class according to each classifier and select the one
        with the highest output probability score.
        4. Calculate and display the fraction of correct predictions over the total number of
        predictions based on the test set.
        5. Plot 3 scatter plots (one for each pair of citizen features) with the dataset and the
        final prediction of the model.
    Returns:
        Nothing.
    """
    df1 = pd.read_csv("../attachments/solar_system_census.csv")
    df2 = pd.read_csv("../attachments/solar_system_census_planets.csv")

    df = pd.concat([df1, df2["Origin"]], axis=1)
    X = np.array(df.iloc[:, 1:4])
    Y = np.array(df["Origin"]).reshape(-1, 1)
    x_train, x_test, y_train_full, y_test_full = data_spliter(X, Y, 0.7)
    xtr = add_polynomial_features(x_train, 3)
    xtst = add_polynomial_features(x_test, 3)
    train_min = xtr.min(axis=0)
    train_max = xtr.max(axis=0)

    x_train_norm = (xtr - train_min) / (train_max - train_min)
    x_test_norm = (xtst - train_min) / (train_max - train_min)

    lambdas_ = np.linspace(0.0, 1.0, num=6)
    scores = {}

    for lambda_ in lambdas_:

        probas = []
        for planet in range(4):
            y_train = (y_train_full == planet).astype(int)

            model = MyLogR(
                thetas=np.ones((xtr.shape[1] + 1, 1)),
                alpha=1e-1,
                max_iter=50000,
                penality="l2",
                lambda_=lambda_,
            ).fit_(x_train_norm, y_train)
            proba = model.predict_(x_test_norm)
            probas.append(proba)

        probas = np.column_stack(probas)
        y_pred = np.argmax(probas, axis=1).reshape(-1, 1)

        score = f1_score_(y_test_full, y_pred)
        scores[lambda_] = score
        print(f"lambda {float(lambda_)} done.")

    print("job finished.")

    key_map = {i: key for i, key in enumerate(scores.keys())}
    for lambda_ in scores.keys():
        print(f"f1_score for lamda_={lambda_}: {scores[lambda_]}")

    predictions = np.argmax(probas, axis=1).reshape(-1, 1)
    best_lambda = key_map.get(np.argmax([score for score in scores.values()]))

    _, ax = plt.subplots(1, 2, figsize=(18, 6))

    for i in range(3):
        x = x_test[:, i]
        ax[0].scatter(
            x, y_test_full, c=predictions.ravel(), alpha=0.6, marker="o", s=100
        )
        ax[0].set_xlabel("Predicted zipcode")
        ax[0].set_ylabel("Target zipcode")
        ax[0].set_title(
            f"Predictions vs Target Values for best model: lambda_={best_lambda}"
        )

    plot_scores = [score * 100 for score in scores.values()]
    bar_colors = ["red", "blue", "green", "grey", "grey", "grey"]

    ax[1].bar(lambdas_, plot_scores, color=bar_colors, width=0.15)
    ax[1].set_xlabel("lambda")
    ax[1].set_ylabel("f1_score in percent (%)")
    ax[1].set_title("f1_score depending on lambda value")

    plt.show()
