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
    data = pd.read_csv('kpi_train.csv', parse_dates=[0])
    page = data[data['event_type']=='$pageview'].set_index('date')[['pv', 'uv']]
    reg = data[data['event_type']=='reg_input_success'].set_index('date')[['pv', 'uv']]
    df = pd.concat([page, reg], axis=1)
    df.columns = ['p_pv', 'p_uv', 'r_pv', 'r_uv']
    data = df.copy()


    # 处理异常数据
    r_abn_date = date(2019, 1, 9) # 周三， 去前后两天你的均值
    r_pv_abn_date = date(2018, 12, 8)
    r_abn_date_pv = np.mean([get_times_data(data, r_abn_date, 2)['r_pv'].values,
                             get_times_data(data, r_abn_date, 2, False)['r_pv'].values])
    data.loc[r_abn_date]['r_pv'] = float_int(r_abn_date_pv)

    r_abn_date_uv = np.mean([get_times_data(data, r_abn_date, 2)['r_uv'].values,
                             get_times_data(data, r_abn_date, 2, False)['r_uv'].values])
    data.loc[r_abn_date]['r_uv'] = float_int(r_abn_date_uv)

    data.loc[r_pv_abn_date]['r_pv'] = data.loc[r_pv_abn_date]['r_uv']

    # 处理节假日
    # 元旦
    yuandan_date_6 = date(2018, 12, 29) # 周六补班， 降， 取前后各一周周六均值
    yuandan_date_7 = date(2018, 12, 30) # 周天假期， 平， 取前后各一周周天均值（受元旦影响）
    yuandan_date_1 = date(2018, 12, 31) # 周一假期， 升， 取前后各一周周一均值
    yuandan_date_2 = date(2019, 1, 1) # 周二假期， 升， 取前后各一周周二均值
    yuandan_date = [yuandan_date_6, yuandan_date_7, yuandan_date_1, yuandan_date_2]
    for dt in yuandan_date:
        temp = (get_times_data_week(data, dt, 1).values
                + get_times_data_week(data, dt, 1, False).values)/2.0
        temp = [float_int(x) for x in temp[0]]

    # 春节七天删除，春节前的周六周天修正， 前一周和后面第二周的均值
    chunjie_6 = date(2019, 2, 2)
    chunjie_7 = date(2019, 2, 3)
    chunjie_date = [chunjie_6, chunjie_7]
    chunjie_drop = pd.date_range('2019-2-4', '2019-2-10')
    data2 = data.drop(index=chunjie_drop)

    for dt in chunjie_date:
        temp = (get_times_data_week(data, dt+timedelta(days=7), 1).values
                + get_times_data_week(data, dt, 1, False).values)/2.0
        temp = [float_int(x) for x in temp[0]]
        # data.loc[dt] = temp

    # 清明节暂且不处理周六周天，修复周五
    qingming_1 = date(2019, 4, 5)
    qingming_date = [qingming_1]
    for dt in qingming_date:
        temp = (get_times_data_week(data, dt, 1).values
                + get_times_data_week(data, dt, 1, False).values)/2.0
        temp = [float_int(x) for x in temp[0]]
        data2.loc[dt]['p_uv'] = temp[1]

    # 五一假期分为三块，前后都补了周末影响了周六，需要修复较多
    wuyi_6_1 = date(2019, 4, 27)
    wuyi_7_1 = date(2019, 4, 28)
    wuyi_6_2 = date(2019, 5, 4)
    wuyi_7_2 = date(2019, 5, 5)
    wuyi_3 = date(2019, 5, 1)
    wuyi_4 = date(2019, 5, 2)
    wuyi_5 = date(2019, 5, 3)
    wuyi_date = [wuyi_6_1, wuyi_7_1, wuyi_6_2, wuyi_7_2, wuyi_3, wuyi_4, wuyi_5]
    for dt in wuyi_date:
        temp = (get_times_data_week(data, dt+timedelta(days=7), 1).values
                + get_times_data_week(data, dt, 1, False).values)/2.0
        temp = [float_int(x) for x in temp[0]]

    data2[['p_pv']][-84:].to_csv('processed_data/p_pv.csv')
    data2[['p_uv']][-98:].to_csv('processed_data/p_uv.csv')
    data2[['r_pv']][-187:].to_csv('processed_data/r_pv.csv')
    data2[['r_uv']][-194:].to_csv('processed_data/r_uv.csv')

    pre_data = pd.DataFrame()
    pre_data['date'] = pd.date_range('2019-5-20', '2019-5-26')
    train = pd.merge(data2, pre_data, how='outer', on='date').reset_index(drop=True)

    user = pd.read_csv('user.csv', parse_dates=['xwhen'], date_parser=date_helper)
    user['day'] = user.xwhen.dt.date
    un = user.groupby('day').size().reset_index()
    un.day = pd.to_datetime(un.day)
    un.columns = ['date', 'un']
    train = pd.merge(train, un, on='date').reset_index(drop=True)

    train.to_csv('processed_data/train_2.csv')


