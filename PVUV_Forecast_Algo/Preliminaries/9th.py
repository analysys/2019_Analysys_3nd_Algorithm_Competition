# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 23:58:49 2019

@author: 41715
"""

import xgboost as xgb

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# %matplotlib widget
# from matplotlib import pyplot as plt

# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

# define a pair  BCH-USDT   LTC-USDT
pd.set_option('expand_frame_repr', False)

#event_file = 'data/event_detail.csv'
kpi_file = 'kpi_train.csv'
# user_file = 'data/user.csv'

kpi_df = pd.read_csv(open(kpi_file))

kpi_df['date'] = pd.to_datetime(kpi_df['date'], format='%Y%m%d')
kpi_df['weekday'] = kpi_df['date'].dt.dayofweek
kpi_df['delta'] = (kpi_df['date'] - pd.to_datetime('2018-11-01')).dt.days

# 4月5日 周五 是清明节，改成节假日 weekday=5
kpi_df.iloc[310:312, 4] = 5

# 修复19年元旦的调休影响  18-12-29 周六 上班  18-12-31 周一放假  19-01-01 元旦
# 把 12-31 与 12-29 weekday数据互换，01-01 作为周日处理 weekday 改成 6
kpi_df.iloc[116, 4] = kpi_df.iloc[117, 4] = 0
kpi_df.iloc[120, 4] = kpi_df.iloc[121, 4] = 5
kpi_df.iloc[122, 4] = 6
kpi_df.iloc[123, 4] = 6

# 修复51调休影响 5-1, 5-2, 5-3 放假，5-5上班
# 4月28号 周日上班 标记 周一
# 5-5 weekday-> 2; 5-1,5-2,5-3 weekday -> 5
kpi_df.iloc[356:358, 4] = 0
kpi_df.iloc[370, 4] = kpi_df.iloc[371, 4] = 2
kpi_df.iloc[362:368, 4] = 5

# pageview 01-14, 01-15 uv 数据异常偏高
kpi_df.drop([148, 150], inplace=True)
# reg_input 12-08 03-02 04-11 偏高 04-27 04-30 偏低
kpi_df.drop([75, 243, 323, 355, 361], inplace=True)

# df.loc['2000-6-1':'2000-6-10']
# 去掉春节时间段 19-02-02 至 19-02-10
mask = (kpi_df['date'] < '2019-2-2') | (kpi_df['date'] > '2019-2-10')
mask_kpi = kpi_df.loc[mask]

mask_kpi = mask_kpi[mask_kpi['event_type'] == 'reg_input_success']
# mask_kpi = mask_kpi[mask_kpi['event_type'] == '$pageview']
print(mask_kpi.head(10))

features = ['weekday', 'delta']
# features = ['event_type', 'weekday', 'delta']
X = mask_kpi[features]
y = mask_kpi.pv
# y = mask_kpi.uv

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# 优化 xgboost 模型参数
from sklearn.model_selection import GridSearchCV
# for tuning parameters
parameters_for_testing = {
    'max_depth':[3, 5, 7],
    'learning_rate':[0.05, 0.01, 0.015],
    'n_estimators':[400, 500, 600],
    'booster': ['gblinear', 'gbtree'],
    'gamma':[0,0.02,0.04],
    'min_child_weight':[1,3,5],
    'subsample':[0.8,0.9,1.0], 
    'colsample_bytree':[0.7,0.8,0.9,1.0],
    'reg_alpha':[0.60, 0.65, 0.70],
    'reg_lambda':[0.85, 0.80, 0.75],
}

xgb_model = xgb.XGBRegressor(max_depth=3, 
                             learning_rate=0.010, 
                             n_estimators=500, 
                             objective='reg:squarederror',
                             booster='gbtree',
                             n_jobs=6, 
                             gamma=0,
                             min_child_weight=1, 
                             subsample=1.0, 
                             colsample_bytree=1.0, 
                             scale_pos_weight=1,
                             random_state=27,
                             reg_alpha=0.60,
                             reg_lambda=0.80
                            )

gsearch1 = GridSearchCV(estimator=xgb_model, param_grid=parameters_for_testing, n_jobs=4, iid=False, cv=5,
                           verbose=10)
gsearch1.fit(X_train, y_train)

# 输出最优模型
print('best params')
print (gsearch1.best_params_)
print('best score')
print (gsearch1.best_score_)
val_predis = gsearch1.predict(X_val)
print(r2_score(y_val, val_predis, multioutput='variance_weighted'))
print(mean_absolute_error(y_val, val_predis))


# 准备预测 X_test, res_df 为pageview 的, res_df2 为 reg_input_success 的
res_df = pd.DataFrame({'date': pd.date_range(start='2019-05-20', end='2019-05-26', freq='1D'), 'event_type': '$pageview'})
res_df['delta'] = (res_df['date'] - pd.to_datetime('2018-11-01')).dt.days
res_df['weekday'] = res_df['date'].dt.dayofweek

res_df2 = pd.DataFrame({'date': pd.date_range(start='2019-05-20', end='2019-05-26', freq='1D'), 'event_type': 'reg_input_success'})
res_df2['delta'] = (res_df2['date'] - pd.to_datetime('2018-11-01')).dt.days
res_df2['weekday'] = res_df2['date'].dt.dayofweek

# 设置预测目标
X_test = res_df2


full_xgb_model = xgb.XGBRegressor(max_depth=3, 
                             learning_rate=0.010, 
                             n_estimators=500, 
                             objective='reg:squarederror',
                             booster='gbtree',
                             n_jobs=6, 
                             gamma=0,
                             min_child_weight=1, 
                             subsample=1.0, 
                             colsample_bytree=1.0, 
                             scale_pos_weight=1,
                             random_state=27,
                             reg_alpha=0.60,
                             reg_lambda=0.80)
full_xgb_model.fit(X, y)
# Calculate the mean absolute error of your Random Forest model on the validation data
val_predis = full_xgb_model.predict(X_test[features])
print(val_predis)

# 预测结果赋值
X_test['pv'] = val_predis

# X_test['uv'] = val_predis


# 合并两个事件的预测结果，生成 .csv 文件
frames = [res_df, res_df2]
res_df3 = pd.concat(frames, sort=True)
print(res_df3)
res_df3['pv'] = res_df3['pv'].round().astype(int)
res_df3['uv'] = res_df3['uv'].round().astype(int)

res_df3['date'] = res_df3['date'].apply(lambda x: x.strftime('%Y%m%d'))
print(res_df3)
res_df3[['date', 'event_type', 'pv', 'uv']].to_csv('xgboost_submission.csv', sep=',', encoding='utf-8', index=False)