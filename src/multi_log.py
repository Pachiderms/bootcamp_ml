from src.data_spliter import data_spliter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.logistic_regression import MyLogisticRegression as MyLR
import ast

def one_vs_all():
    df1 = pd.read_csv("../attachments/solar_system_census.csv")
    df2 = pd.read_csv("../attachments/solar_system_census_planets.csv")


    df = pd.concat([df1, df2['Origin']], axis=1)
    X = np.array(df.iloc[:, 1:4])
    models = []
    y = []
    y_hats = []

    for i in range(4):
        d = df.copy()
        d['Origin'] = np.where(df['Origin'] == i, 1, 0)
        Y = np.array(d['Origin']).reshape(-1, 1)
    
        (x_train, x_test, y_train, y_test) = data_spliter(X, Y, 0.8)
        y.append(y_test)
        train_min = x_train.min(axis=0)
        train_max = x_train.max(axis=0)
        
        x_train_norm = (x_train - train_min) / (train_max - train_min)
        x_test_norm = (x_test - train_min) / (train_max - train_min)

        model = MyLR(thetas=np.ones((x_train.shape[1] + 1, 1)), alpha=1e-1, max_iter=50000).fit_(x_train_norm, y_train)
        models.append(model)

        y_hat = model.predict_(x_test_norm)
        y_hats.append(y_hat)

    out = pd.DataFrame(data={
        'planet0': [y_hats[0].tolist()],
        'planet1': [y_hats[1].tolist()],
        'planet2': [y_hats[2].tolist()],
        'planet3': [y_hats[3].tolist()],
    })

    cols = out.columns

    for id, row in out.iterrows():
        arr = np.column_stack([row[c] for c in cols])

        max_idx = np.argmax(arr, axis=1) 

        winners = np.zeros_like(arr, dtype=int)
        winners[np.arange(arr.shape[0]), max_idx] = 1

        for i, col in enumerate(cols):
            out.at[id, col] = winners[:, i].tolist()

    predictions = [out['planet0'], out['planet1'], out['planet2'], out['planet3']]
    print(type(predictions))
