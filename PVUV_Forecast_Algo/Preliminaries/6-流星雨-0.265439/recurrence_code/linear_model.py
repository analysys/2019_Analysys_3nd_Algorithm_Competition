import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge


def prepare_dataset(data, i, col, is_train=True):
    dd = data[col]
    X = pd.DataFrame({
        'l_7': [dd[i-7]],
        "l_14": [dd[i-14]],
        "l_7_7": [dd[i-7]-dd[i-14]],
        "l_14_7": [dd[i-7]-dd[i-21]],
        "mean_1": [(dd[i-7] + dd[i-14])/2],
        "dow": [data.date.dt.dayofweek[i]],
        "un": [data['un'][i]]
    })

    if is_train:
        y = dd[i]
        return X, y
    return X

def model_fit_predict(col):

    data = pd.read_csv('processed_data/train_2.csv', parse_dates=['date'])
    X_l, y_l = [], []
    for i in range(21, 193):
        X_tmp, y_tmp = prepare_dataset(data, i, col)
        X_l.append(X_tmp)
        y_l.append(y_tmp)
    X_train = pd.concat(X_l, axis=0)
    y_train = y_l

    test_x = []
    for i in range(193, 200):
        X_tmp = prepare_dataset(data, i, col, is_train=False)
        test_x.append(X_tmp)
    X_test = pd.concat(test_x, axis=0)

    train_y=y_train
    clf = Ridge(alpha=0.001, normalize=True)
    clf.fit(X_train, train_y)
    predict_y = clf.predict(X_test)

    return predict_y
