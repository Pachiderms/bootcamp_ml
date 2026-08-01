from src.data_spliter import data_spliter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.logistic_regression import MyLogisticRegression as MyLR

def mono_log_(zipcode=0):
    print(f"trainin on {zipcode=}")
    df1 = pd.read_csv("../attachments/solar_system_census.csv")
    df2 = pd.read_csv("../attachments/solar_system_census_planets.csv")

    df = pd.concat([df1, df2['Origin']], axis=1)

    X = np.array(df.iloc[:, 1:4])
    Y = np.array(df['Origin']).reshape(-1, 1)

    (x_train, x_test, y_train_full, y_test_full) = data_spliter(X, Y, 0.8)
    y_train = (y_train_full == zipcode).astype(int)
    y_test = (y_test_full == zipcode).astype(int)
    train_min = x_train.min(axis=0)
    train_max = x_train.max(axis=0)

    x_train_norm = (x_train - train_min) / (train_max - train_min)
    x_test_norm = (x_test - train_min) / (train_max - train_min)

    my_log_reg = MyLR(thetas=np.ones((x_train.shape[1] + 1, 1)), alpha=1e-1, max_iter=50000).fit_(x_train_norm, y_train)

    prob = my_log_reg.predict_(x_test_norm)
    prediction = prob.copy()
    prediction[prediction > .5] = 1.0
    prediction[prediction <= .5] = 0.0

    match = y_test == prediction
    print(f"eval predictions: {np.sum(match) * 100 / len(y_test)}% accurate.\n\
({np.sum(match)} good predictions out of {len(y_test)}).")

    weight = x_test[:, 0]
    height = x_test[:, 1]
    bone_density = x_test[:, 2]

    fig, axs = plt.subplots(1, 3, figsize=(20, 10))

    axs[0].scatter(weight, height, c=y_test.ravel())
    axs[0].scatter(weight, height, c=prediction.ravel(), marker='x')
    axs[0].set_xlabel("Weight")
    axs[0].set_ylabel("Height")
    axs[0].set_title("Height vs Weight")

    axs[1].scatter(weight, bone_density, c=y_test.ravel())
    axs[1].scatter(weight, bone_density, c=prediction.ravel(), marker='x')
    axs[1].set_xlabel("Weight")
    axs[1].set_ylabel("Bone Density")
    axs[1].set_title("Weight vs Bone Density")

    axs[2].scatter(height, bone_density, c=y_test.ravel())
    axs[2].scatter(height, bone_density, c=prediction.ravel(), marker='x')
    axs[2].set_xlabel("Height")
    axs[2].set_ylabel("Bone Density")
    axs[2].set_title("Height vs Bone Density")

    plt.show()