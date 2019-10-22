from datetime import date, timedelta
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error



def get_nearwd(date, b_date):
    date_list = pd.date_range(date - timedelta(140), periods=21, freq='7D').date
    result = date_list[date_list <= b_date][-1]
    return result

def get_timespan(df, dt, minus, periods, freq='D'):
    return df.loc[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]


def prepare_dataset(t2018, is_train=True):
    # 前几天的和
    X = pd.DataFrame({
        "sum_7": get_timespan(df_train, t2018, 7, 7).sum().values,
        "sum_14": get_timespan(df_train, t2018, 14, 14).sum().values,
    })

    # 前一周中每天的特征
    for i in range(7):
        #  前07/14/28。。。天的均值
        X['mean_4_dow_{}'.format(i)] = get_timespan(df_train, t2018, 28 - i, 4, freq='7D').mean().values
        # X['mean_12_dow{}_2017'.format(i)] = get_timespan(df_train, t2018, 84-i, 12, freq='7D').mean(axis=1).values
        X['mean_20_dow_{}'.format(i)] = get_timespan(df_train, t2018, 140 - i, 20, freq='7D').mean().values

        new_date = get_nearwd(t2018 + timedelta(i), t2018)
        ahead = (t2018 - new_date).days
        if ahead != 0:
            # ahead n、n+7 天的均值
            X['ahead0_{}'.format(i)] = get_timespan(df_train, new_date + timedelta(ahead), ahead, ahead).mean().values
            X['ahead7_{}'.format(i)] = get_timespan(df_train, new_date + timedelta(ahead), ahead + 7, ahead + 7).mean().values
        # ahead n、n+7
        X["day_1_2017_{}_1".format(i)] = get_timespan(df_train, new_date, 1, 1).values.ravel()
        X["day_1_2017_{}_2".format(i)] = get_timespan(df_train, new_date - timedelta(7), 1, 1).values.ravel()
        # 3 7 14 30 60 140 天的均值
        for m in [3, 7, 14, 21, 30, 60, 140]:
            X["mean_{}_2017_{}_1".format(m, i)] = get_timespan(df_train, new_date, m, m).mean().values
            X["mean_{}_2017_{}_2".format(m, i)] = get_timespan(df_train, new_date - timedelta(7), m, m).mean().values
    if is_train:
        y = df_train.loc[
            pd.date_range(t2018, periods=7)
        ].T.values
        return X, y
    return X
def lightgbm_model_fit():
    data = pd.read_csv('kpi_train.csv')
    df_page = data[data['event_type']=='$pageview'][['pv', 'uv']]\
        .rename(columns={'pv':'page_pv', 'uv':'page_uv'}).reset_index(drop=True)
    df_reg =  data[data['event_type']=='reg_input_success'][['pv', 'uv']]\
        .rename(columns={'pv':'reg_pv', 'uv':'reg_uv'}).reset_index(drop=True)
    df_train = pd.concat([df_page, df_reg], axis=1)
    df_train.index = pd.date_range('20181101', '20190519')
    df_train = df_train[['page_pv', 'page_uv']]
    t2018 = date(2018, 11, 18)
    X_l, y_l = [], []
    for i in range(25*7+1): # 26
        delta = timedelta(days= i)
        X_tmp, y_tmp = prepare_dataset(
            t2018 + delta
        )
        X_l.append(X_tmp)
        y_l.append(y_tmp)
    X_train = pd.concat(X_l, axis=0)
    y_train = np.concatenate(y_l, axis=0)

    del X_l, y_l
    X_val, y_val = prepare_dataset(date(2019, 5, 5))
    X_test = prepare_dataset(date(2019, 5, 19), is_train=False)

    print("Training and predicting models...")
    params = {
        'num_leaves': 16,
        'objective': 'regression',
        # 'min_data_in_leaf': 3,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.85,
        'bagging_freq': 3,
        'metric': 'l2_root',
        'num_threads': 4
    }

    MAX_ROUNDS = 200
    val_pred = []
    test_pred = []
    cate_vars = []
    for i in range(7):
        print("=" * 50)
        print("Step %d" % (i + 1))
        print("=" * 50)
        dtrain = lgb.Dataset(
            X_train, label=y_train[:, i],
            # categorical_feature=cate_vars,
        )
        dval = lgb.Dataset(
            X_val, label=y_val[:, i], reference=dtrain,)
            # categorical_feature=cate_vars)
        bst = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=50
        )
        print("\n".join(("%s: %.2f" % x) for x in sorted(
            zip(X_train.columns, bst.feature_importance("gain")),
            key=lambda x: x[1], reverse=True
        )))
        val_pred.append(bst.predict(
            X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
        test_pred.append(bst.predict(
            X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))

    print("Validation mse:", mean_squared_error(
        y_val, np.array(val_pred).transpose()) ** 0.5)

    print("Making submission...")
    y_test = np.array(test_pred).transpose()
    df_preds = pd.DataFrame(y_test).apply(lambda row : [int(x)+1 if x-int(x)>0.5 else int(x) for x in row])
    # df_preds.T.to_csv('lgb_1.csv', index=False)