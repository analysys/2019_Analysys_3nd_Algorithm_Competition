import numpy as np
import pandas as pd
from tbats import TBATS
from data_process import data_process

if __name__ == "__main__":

    process_data = data_process()

    tbats_res = []

    p_pv = []
    temp = np.array(process_data[0])[0]
    estimator = TBATS(seasonal_periods=[7])
    fitted_model = estimator.fit(temp)
    y_1 = fitted_model.forecast(steps=7)
    temp = np.array(process_data[1])[0]
    estimator = TBATS(seasonal_periods=[7])
    fitted_model = estimator.fit(temp)
    y_2 = fitted_model.forecast(steps=7)
    for i in range(5):
        p_pv.append(0.65*y_1[i]+0.35*y_2[i])
    p_6 = (temp[-2] + temp[-9])*0.5
    p_7 = (temp[-1] + temp[-8])*0.5
    p_pv.append(p_6)
    p_pv.append(p_7)
    tbats_res.append(p_pv)

    p_uv = []
    temp = np.array(process_data[2])[0]
    estimator = TBATS(seasonal_periods=[7])
    fitted_model = estimator.fit(temp)
    y_1 = fitted_model.forecast(steps=7)
    temp = np.array(process_data[3])[0]
    estimator = TBATS(seasonal_periods=[7])
    fitted_model = estimator.fit(temp)
    y_2 = fitted_model.forecast(steps=7)
    for i in range(5):
        p_uv.append(0.6*y_1[i]+0.4*y_2[i])
    p_6 = (temp[-2] + temp[-9])*0.5
    p_7 = (temp[-1] + temp[-8])*0.5
    p_uv.append(p_6)
    p_uv.append(p_7)
    tbats_res.append(p_uv)

    r_pv = []
    temp = np.array(process_data[4])[0]
    for i in range(7):
        r_pv.append(temp[i-21]*0.2+temp[i-14]*0.4+temp[i-7]*0.4)
    tbats_res.append(r_pv)

    r_uv = []
    temp = np.array(process_data[5])[0]
    for i in range(7):
        r_uv.append(temp[i - 21] * 0.2 + temp[i - 14] * 0.4 + temp[i - 7] * 0.4)
    tbats_res.append(r_uv)

    res = pd.DataFrame(tbats_res).T
    res.columns = ['p_pv', 'p_uv', 'r_pv', 'r_uv']

    # 编辑成提交格式
    res_df= res.apply(lambda row: [int(x) + 1 if x - int(x) > 0.5 else int(x) for x in row])
    res_page = pd.DataFrame()
    res_page['date'] = ['20190826', '20190827', '20190828', '20190829', '20190830', '20190831', '20190901']
    res_page['event_type'] = ['$pageview'] * 7
    res_page['pv'] = res_df['p_pv']
    res_page['uv'] = res_df['p_uv']


    res_reg = pd.DataFrame()
    res_reg['date'] = ['20190826', '20190827', '20190828', '20190829', '20190830', '20190831', '20190901']
    res_reg['event_type'] = ['reg_input_success'] * 7
    res_reg['pv'] = res_df['r_pv']
    res_reg['uv'] = res_df['r_uv']


    res = pd.concat([res_page, res_reg])

    res.to_csv('final_result.csv', index=False)