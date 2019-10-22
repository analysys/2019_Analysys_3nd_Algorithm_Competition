#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入必要的库
from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import  stats
import matplotlib.pyplot as plt
from fbprophet import Prophet


# In[2]:


# 从数据集统计周期因子
train = pd.read_csv('kpi_train.csv')
train.columns = ['Date', 'event_type', 'pv', 'uv']
train_page = train[train['event_type'] == '$pageview'].reset_index()
train_reg = train[train['event_type'] == 'reg_input_success'].reset_index()
del train
train = pd.DataFrame([[0] * 5] * 200)
train.columns = ['Date', 'page_pv', 'page_uv', 'reg_pv', 'reg_uv']
train['Date'] = train_page['Date']
train['page_pv'] = train_page['pv']
train['page_uv'] = train_page['uv']
train['reg_pv'] = train_reg['pv']
train['reg_uv'] = train_reg['uv']

train["Date"] = pd.to_datetime(train["Date"], format='%Y%m%d')
train["year"] = train["Date"].dt.year.astype('float64')
train["month"] = train["Date"].dt.month.astype('float64')
train["date"] = train["Date"].dt.day.astype('float64')
train["day"] = train["Date"].dt.dayofweek.astype('float64')
train = train[:][4:]
train.reset_index(inplace=True)
def compute_weekly_mean(value_list):
    temp_mean = []
    for index in range(int(len(value_list) / 7)):
        temp_mean.extend([sum(value_list[index * 7: (index + 1) * 7]) / 7] * 7)
    return temp_mean


train['page_pv_mean'] = compute_weekly_mean(train['page_pv'].tolist())
train['page_uv_mean'] = compute_weekly_mean(train['page_uv'].tolist())
train['reg_pv_mean'] = compute_weekly_mean(train['reg_pv'].tolist())
train['reg_uv_mean'] = compute_weekly_mean(train['reg_uv'].tolist())

weekly_mean = []
weekly_mean.append(compute_weekly_mean(train['page_pv'].to_list()))
weekly_mean.append(compute_weekly_mean(train['page_uv'].to_list()))
weekly_mean.append(compute_weekly_mean(train['reg_pv'].to_list()))
weekly_mean.append(compute_weekly_mean(train['reg_uv'].to_list()))

train['page_pv_factor'] = train['page_pv'] / train['page_pv_mean']
train['page_uv_factor'] = train['page_uv'] / train['page_uv_mean']
train['reg_pv_factor'] = train['reg_pv'] / train['reg_pv_mean']
train['reg_uv_factor'] = train['reg_uv'] / train['reg_uv_mean']
train.to_csv('train.csv', index=False)


# In[3]:


train[['page_pv_factor', 'page_uv_factor', 'reg_pv_factor', 'reg_uv_factor']].tail()


# In[4]:


train['page_pv_factor'].iloc[[28,190,93,115,74,194,97]]  # 周期因子


# In[5]:


#节假日设置
playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2019-04-05', '2019-04-06','2019-04-07',
                       '2019-05-01','2019-05-02','2019-05-03',
                       '2019-05-04']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2019-05-05']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))


# In[7]:


##################################################################################
# 注意：此处事先用excel对日期数据进行批量，例如：将‘20180101’改为‘2018-01-01’#
#################################################################################
# 设定数据位置
RAW_DATA = 'kpi_train.csv'
df = pd.read_csv(RAW_DATA)
df['date'] = pd.to_datetime(df['date'], format = '%Y%m%d')
df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
df.to_csv('kpi_train_1.csv')

RAW_DATA = 'kpi_train_1.csv'
# 读取pv数据

star_point = 200


df_pv = pd.read_csv(RAW_DATA,usecols=['pv','date'])
df_pv = df_pv.rename(columns={'date':'ds', 'pv':'y'})
dta1 = df_pv[star_point:400:2]
dta2 = df_pv[star_point+1:401:2]


# In[8]:


# 建立prophet模型
m = Prophet(changepoint_prior_scale=0.03,weekly_seasonality=True,interval_width=0.95,holidays=holidays,holidays_prior_scale=2.0,changepoint_range=0.95)
m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.1)
# 拟合数据
m.fit(dta1)
# 建立预测范围
future = m.make_future_dataframe(periods=7)
# 预测数据集
forecast = m.predict(future)
dta1_predict = forecast[[ 'yhat']].tail(7)
dta1_predict *= np.array(train['page_pv_factor'].iloc[[28,190,117,115,74,194,97]]).reshape(7,1)
# 画出预测数据折线图
#m.plot(forecast)
print(dta1_predict)


# In[9]:


# 建立prophet模型
m = Prophet(changepoint_prior_scale=0.04,weekly_seasonality=True,interval_width=0.95,holidays=holidays,holidays_prior_scale=20.0,changepoint_range=0.78)
m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.1)
# 拟合数据
m.fit(dta2)
# 建立预测范围
future = m.make_future_dataframe(periods=7)
# 预测数据集
forecast = m.predict(future)
dta2_predict = forecast[[ 'yhat']].tail(7)
dta2_predict *= np.array(train['reg_pv_factor'].iloc[[126,29, 37,10, 25,194,13]]).reshape(7, 1)
# 画出预测数据折线图
#m.plot(forecast)
print(dta2_predict)


# In[10]:


# 读取uv数据
df_uv = pd.read_csv(RAW_DATA,usecols=['uv','date'])
df_uv = df_uv.rename(columns={'date':'ds', 'uv':'y'})
dta3 = df_uv[star_point:400:2]
dta4 = df_uv[star_point+1:401:2]


# In[11]:


# 建立prophet模型
m = Prophet(changepoint_prior_scale=0.07,weekly_seasonality=True,interval_width=0.95,holidays=holidays,holidays_prior_scale=10.0,changepoint_range=0.72)
m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.1)
# 拟合数据
m.fit(dta3)
# 建立预测范围
future = m.make_future_dataframe(periods=7)
# 预测数据集
forecast = m.predict(future)
dta3_predict = forecast[[ 'yhat']].tail(7)
dta3_predict *= np.array(train['page_uv_factor'].iloc[[154,106,93,52,67,61,97]]).reshape(7,1)
# 画出预测数据折线图
#m.plot(forecast)
print(dta3_predict)


# In[12]:


# 建立prophet模型
m = Prophet(changepoint_prior_scale=0.06,weekly_seasonality=True,interval_width=0.95,holidays=holidays, holidays_prior_scale=10.0,changepoint_range=0.72)
m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.1)
# 拟合数据 
m.fit(dta4)
# 建立预测范围
future = m.make_future_dataframe(periods=7)
# 预测数据集
forecast = m.predict(future)
dta4_predict = forecast[[ 'yhat']].tail(7)
dta4_predict *= np.array(train['reg_uv_factor'].iloc[[112, 78, 72, 101, 25, 19, 48]]).reshape(7,1)

# 画出预测数据折线图
#m.plot(forecast)
print(dta4_predict)


# In[13]:


# 将预测值转化为整数
for i in range(7):
    dta1_predict.values[i] = np.round(dta1_predict.values[i])
    dta2_predict.values[i] = np.round(dta2_predict.values[i])
    dta3_predict.values[i] = np.round(dta3_predict.values[i])
    dta4_predict.values[i] = np.round(dta4_predict.values[i])
# 构建写入csv文件的数组
pv_predict = np.zeros(14)
uv_predict = np.zeros(14)

for i in range(14):
    if i%2 == 0:
        pv_predict[i] = dta1_predict.values[int(i/2)]
        uv_predict[i] = dta3_predict.values[int(i/2)]
    if i%2 == 1:
        pv_predict[i] = dta2_predict.values[int(i/2)]
        uv_predict[i] = dta4_predict.values[int(i/2)]
pv_predict = pv_predict.astype(np.int32)
uv_predict = uv_predict.astype(np.int32)
#print(type(pv_predict[1]))
#print(pv_predict.dtype)


# In[14]:


# 导入必要的库
import csv
# 设置存储位置
PREDICT_DATA = 'facebook_prophet_and_time_series_factor_model_{}.csv'.format(star_point)
with open(PREDICT_DATA,"w",newline='') as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerow(["date","event_type","pv","uv"])
    date = ['20190520','20190520','20190521','20190521','20190522','20190522','20190523','20190523','20190524','20190524','20190525','20190525','20190526','20190526','20190527','20190527']
    etype = ['$pageview','reg_input_success','$pageview','reg_input_success','$pageview','reg_input_success','$pageview','reg_input_success','$pageview','reg_input_success','$pageview','reg_input_success','$pageview','reg_input_success']
    for i in range(14):
        writer.writerow([date[i],etype[i],pv_predict[i],uv_predict[i]])


# In[ ]:




