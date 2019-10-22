import numpy as np
import pandas as pd
from tbats import TBATS
from params import params_0
from linear_model import model_fit_predict
from data_process import data_process

if __name__ == "__main__":
    # 数据预处理
    data_process()

    data_dir = ['p_pv', 'p_uv', 'r_pv', 'r_uv']

    # 线性全局模型
    linear_res = []
    for idx, name in enumerate(data_dir):
        y_forecasted = model_fit_predict(name)
        linear_res.append(y_forecasted)
    linear_res = pd.DataFrame(list(linear_res)).T
    linear_res.columns = ['p_pv', 'p_uv', 'r_pv', 'r_uv']

    # 传统分解模型
    tbats_res = []
    for idx, name in enumerate(data_dir):
        data = pd.read_csv('processed_data/' + name + '.csv')[[name]].T
        print(idx)
        data = np.array(data)[0]
        estimator = TBATS(seasonal_periods=params_0['seasonal_periods'][idx])
        fitted_model = estimator.fit(data)
        y_forecasted = fitted_model.forecast(steps=7)
        y_forecasted = [x * params_0['after_rate'][idx] for x in y_forecasted]
        tbats_res.append(y_forecasted)

    tbats_res = pd.DataFrame(tbats_res).T
    tbats_res.columns = ['p_pv', 'p_uv', 'r_pv', 'r_uv']

    # 模型融合
    rate = params_0['rh_rate']
    res = pd.DataFrame()
    res['p_pv'] = linear_res['p_pv'].values * rate + tbats_res['p_pv'].values * (1 - rate)
    res['p_uv'] = linear_res['p_uv'].values * rate + tbats_res['p_uv'].values * (1 - rate)
    res['r_pv'] = tbats_res['r_pv']
    res['r_uv'] = tbats_res['r_uv']

    # 编辑成提交格式
    res_df= res.apply(lambda row: [int(x) + 1 if x - int(x) > 0.5 else int(x) for x in row])
    res_page = pd.DataFrame()
    res_page['date'] = ['20190520', '20190521', '20190522', '20190523', '20190524', '20190525', '20190526']
    res_page['event_type'] = ['$pageview'] * 7
    res_page['pv'] = res_df['p_pv']
    res_page['uv'] = res_df['p_uv']


    res_reg = pd.DataFrame()
    res_reg['date'] = ['20190520', '20190521', '20190522', '20190523', '20190524', '20190525', '20190526']
    res_reg['event_type'] = ['reg_input_success'] * 7
    res_reg['pv'] = res_df['r_pv']
    res_reg['uv'] = res_df['r_uv']

    w = list(res_reg['pv'] - res_reg['uv'] > 10)
    res_reg['uv'][w] = res_reg['pv'][w] - \
                       (res_reg['pv'][w.index(False)] - res_reg['uv'][w.index(False)])

    res = pd.concat([res_page, res_reg])

    res.to_csv('F:/Eguan/第二届是算法大赛/源码/6-流星雨-0.265439/recurrence_code/final_result.csv', index=False)