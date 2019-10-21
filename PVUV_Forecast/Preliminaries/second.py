# %%
import pandas as pd
from datetime import date
import numpy as np
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
# %% read training data


def read_data():
    print("Preparing dataset...")
    y_train = pd.read_csv(
        'kpi_train.csv',
        parse_dates=["date"]
    ).set_index(['date']).groupby('event_type')

    y_train1 = y_train.get_group(
        'reg_input_success').drop('event_type', axis=1)
    y_train2 = y_train.get_group('$pageview').drop('event_type', axis=1)
    y_train = pd.merge(y_train2, y_train1,
                       left_index=True, right_index=True)

    y = y_train[['pv_x', 'uv_x', 'pv_y', 'uv_y']]
    return y


def get_wave(y, tau=7, N=10):
    z = np.arange(len(y.index))
    Ua = y.values

    def fourier(x, *a):
        ret = a[0] * np.cos(np.pi*2 / tau * x) #傅里叶级数
        for deg in range(1, len(a)):
            ret += a[deg] * np.cos((deg+1) * np.pi * 2 / tau * x)
        return ret
    popt, pcov = curve_fit(fourier, z, Ua, [1.0] * N)
    return fourier(np.arange(len(y.index) + 7), *popt)


def augFeatures(train):
    # use date as features
    train["Date"] = pd.to_datetime(train.index)
    train["year"] = train["Date"].dt.year
    train["month"] = train["Date"].dt.month
    train["day"] = train["Date"].dt.dayofweek
    train["Date"] = train["Date"].dt.day

    return train


def holidays_data():
    holidays = pd.DataFrame(index=pd.date_range(
        date(2018, 11, 1), date(2019, 5, 26)))
    holidays['holidays'] = 0
    holidays.loc[pd.date_range(date(2018, 12, 30), date(2019, 1, 1))] = 1
    holidays.loc[pd.date_range(date(2019, 2, 2), date(2019, 2, 3))] = 2
    holidays.loc[pd.date_range(date(2019, 2, 4), date(2019, 2, 10))] = 1
    holidays.loc[pd.date_range(date(2019, 4, 5), date(2019, 4, 7))] = 1
    holidays.loc[pd.date_range(date(2019, 5, 1), date(2019, 5, 4))] = 1
    holidays.loc[date(2019, 4, 28)] = 2
    holidays.loc[date(2019, 5, 5)] = 2
    return holidays


def submission(fitted):
    df = pd.read_csv('kpi_train.csv', index_col=['date'])
    df = df.iloc[-14:]
    df.index = df.index[:] + 7
    df.loc[df['event_type'] == '$pageview', [
        'pv', 'uv']] = fitted[[0, 1], :].transpose()
    df.loc[df['event_type'] == 'reg_input_success', [
        'pv', 'uv']] = fitted[[2, 3], :].transpose()
    df.to_csv('submission.csv')
    print(df)
    pass

# %%


def learning(n_item, parameters, uselog=1):
    # parameters list is here
    # parameters = {
    #     'week_n': ,
    #     'halfmonth_n': ,
    #     'month_n': ,
    #     'season_n': ,
    #     'year_n': ,
    #     'num_leaves': ,
    #     'min_data_in_leaf':,
    #     'learning_rate': ,
    #     'feature_fraction': ,
    #     'bagging_fraction': ,
    #     'bagging_freq': ,
    # }

    y = read_data()

    week_n = parameters['week_n']
    halfmonth_n = parameters['halfmonth_n']
    month_n = parameters['month_n']
    season_n = parameters['season_n']
    year_n = parameters['year_n']

    x = pd.DataFrame(holidays_data())

    print("Creating features...")
    for j in range(4):
        tmp = y
        if (j > 1) * uselog:
            tmp = np.log(y)
        x['week_wave_%d' % j] = get_wave(
            tmp.iloc[:, j], 7, week_n)
        x['halfmonth_wave_%d' % j] = get_wave(
            tmp.iloc[:, j], 15.21875, halfmonth_n)
        x['month_wave_%d' % j] = get_wave(
            tmp.iloc[:, j], 30.4375, month_n)
        x['season_wave_%d' % j] = get_wave(
            tmp.iloc[:, j], 91.3125, season_n)
        x['year_wave_%d' % j] = get_wave(
            tmp.iloc[:, j], 365.25, year_n)
    augFeatures(x)

    scaler_var = MinMaxScaler(feature_range=(0, 1))
    x.iloc[:, :] = scaler_var.fit_transform(x.iloc[:, :].values)

    params_lgbm = {
        'num_leaves': parameters['num_leaves'],
        'objective': 'regression',
        'min_data_in_leaf': parameters['min_data_in_leaf'],
        'learning_rate': parameters['learning_rate'],
        'feature_fraction': parameters['feature_fraction'],
        'bagging_fraction': parameters['bagging_fraction'],
        'bagging_freq': parameters['bagging_freq'],
        'metric': 'mape',
        'num_threads': parameters['num_threads'],
        # 'device': 'gpu',
    }

    j = n_item

    MAX_ROUNDS = 5000
    splitday = parameters['splitday']
    training_range = pd.date_range(
        date(2018, 11, 1), date(2019, 5, splitday), freq='D')
    val_range = pd.date_range(
        date(2019, 5, 12), date(2019, 5, 19), freq='D')

    dtrain = lgb.Dataset(
        x.loc[training_range],
        label=y.iloc[:, j][training_range])
    dval = lgb.Dataset(
        x.loc[val_range],
        label=y.iloc[:, j][val_range], reference=dtrain)
    bst = lgb.train(
        params_lgbm, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=125, verbose_eval=-1)

    y_pred = bst.predict(x['2019-05-20':])
    y_val = bst.predict(x.loc[val_range])
    print(y_val - y.iloc[:, j][val_range].values)

    return y_pred


def correct_prediction(pred):
    for i in [3]:
        if sum(pred[i] < pred[i - 1]) < 7:
            idx = pred[i] > pred[i - 1]
            a = pred[i][idx]
            b = pred[i - 1][idx]
            c = ((a - b) / 2 + 1).round()
            a = a - c
            b = b + c
            pred[i][idx] = a
            pred[i - 1][idx] = b
            if sum(pred[i] <= pred[i - 1]) < 7:
                pred = correct_prediction(pred)
    if sum(pred[2] - pred[3] > 5) > 0:
        idx = pred[2] - pred[3] > 5
        pred[2, idx] = pred[3, idx] + 1

    # increase top 
    s = pred[0] + pred[1]
    for i in range(2):
        idx = np.argmax(s)
        pred[0, idx] = 1.1 * pred[0, idx]
        s[idx] = 0
    s = pred[3] + pred[2]
    for i in range(1):
        idx = np.argmax(s)
        pred[2:4, idx] = 1.1 * pred[2:4, idx]
        s[idx] = 0

    return pred


# %%
if __name__ == "__main__":

    y_pred = []


    parameters0 = {'week_n': 36, 'halfmonth_n': 18, 'month_n': 15, 'season_n': 21, 'year_n': 48, 'num_leaves': 10, 'min_data_in_leaf': 3,
                  'learning_rate': 0.049, 'feature_fraction': 0.385, 'bagging_fraction': 0.874, 'bagging_freq': 30, 'num_threads': 6, 'splitday': 11}
    y_pred.append(learning(0, parameters0))

    parameters1 = {'week_n': 29, 'halfmonth_n': 11, 'month_n': 18, 'season_n': 16, 'year_n': 44, 'num_leaves': 5, 'min_data_in_leaf': 2, 'learning_rate': 0.04329969894951145, 'feature_fraction': 0.32821992709979714, 'bagging_fraction': 0.42865849197650663, 'bagging_freq': 27, 'num_threads': 16, 'splitday': 11}

    y_pred.append(learning(1, parameters1))

    parameters2 = {'week_n': 34, 'halfmonth_n': 15, 'month_n': 45, 'season_n': 16, 'year_n': 23, 'num_leaves': 16, 'min_data_in_leaf': 1, 'learning_rate': 0.0009874257600165207, 'feature_fraction': 0.713938187034955, 'bagging_fraction': 0.5903210925010334, 'bagging_freq': 25, 'num_threads': 1, 'splitday': 19}

    y_pred.append(learning(2, parameters2, 0))

    parameters3 = {'week_n': 40, 'halfmonth_n': 16, 'month_n': 41, 'season_n': 14, 'year_n': 25, 'num_leaves': 15, 'min_data_in_leaf': 6, 'learning_rate': 0.0005242593134812447, 'feature_fraction': 0.8536485147970311, 'bagging_fraction': 0.8070872371014869, 'bagging_freq': 26, 'num_threads': 16, 'splitday': 19}

    y_pred.append(learning(3, parameters3))

    fitted = np.array(y_pred).reshape([4, 7]).round().astype(int)
    fitted = correct_prediction(fitted.round()).astype(int)
    # print(fitted)
    submission(fitted)

    pass
