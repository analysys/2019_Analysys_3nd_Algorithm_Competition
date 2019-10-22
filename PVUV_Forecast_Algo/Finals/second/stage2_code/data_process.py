from datetime import date, timedelta
import numpy as np
import pandas as pd


def get_times_data(df, dt, periods, back=True):
    if back is True:
        return df.loc[pd.date_range(dt+timedelta(days=1), periods=periods, freq='D')]
    else:
        return df.loc[pd.date_range(dt-timedelta(days=periods), periods=periods, freq='D')]


def get_times_data_week(df, dt, periods, back=True):
    if back is True:
        return df.loc[pd.date_range(dt+timedelta(days=7), periods=periods, freq='7D')]
    else:
        return df.loc[pd.date_range(dt-timedelta(days=periods*7), periods=periods, freq='7D')]


def date_helper(i):
    return pd.to_datetime(i, unit='ms')


def float_int(f):
    if f - int(f) >= 0.5:
        return int(f) + 1
    return int(f)

def data_process():
    data = pd.read_csv('kpi_train_new.csv', parse_dates=[0])
    page = data[data['event_type']=='$pageview'].set_index('date')[['pv', 'uv']]
    reg = data[data['event_type']=='reg_input_success'].set_index('date')[['pv', 'uv']]
    df = pd.concat([page, reg], axis=1)
    df.columns = ['p_pv', 'p_uv', 'r_pv', 'r_uv']
    data = df.copy()


    # 处理异常数据

    dacu_drop = pd.date_range('2019-6-17', '2019-8-4')
    data2 = data.drop(index=dacu_drop)

    dt = date(2019, 8, 16)
    data.loc[dt]['r_pv'] = 100
    data.loc[dt]['r_uv'] = 100

    dt = date(2019, 8, 25)
    data.loc[dt]['r_pv'] = 23
    data.loc[dt]['r_uv'] = 23

    return data2[['p_pv']].T, data2[['p_pv']][228:].T, \
           data2[['p_uv']].T, data2[['p_uv']][228:].T, \
           data[['r_pv']][228:].T, \
           data[['r_uv']][228:].T
    # data2[['p_pv']].to_csv('processed_data/p_pv.csv')
    # data2[['p_pv']][228:].to_csv('processed_data/p_pv_228.csv')
    # data2[['p_uv']].to_csv('processed_data/p_uv.csv')
    # data2[['p_uv']][228:].to_csv('processed_data/p_uv_228.csv')
    # data[['r_pv']][228:].to_csv('processed_data/r_pv_228.csv')
    # data[['r_uv']][228:].to_csv('processed_data/r_uv_228.csv')


