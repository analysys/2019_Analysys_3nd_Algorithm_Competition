# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 23:39:18 2019

@author: 41715
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
%matplotlib inline
from matplotlib.collections import LineCollection
import time

data_event = pd.read_csv('event_detail.csv')
data_train = pd.read_csv('kpi_train.csv')
data_user = pd.read_csv('user.csv')

#首先我们先对数据进行拆分， 将 $pageview和reg_input_success，pv和uv分开
pageview_pv = data_train[data_train['event_type']=='$pageview'][['date','pv']]
pageview_uv = data_train[data_train['event_type']=='$pageview'][['date','uv']]
reg_input_pv = data_train[data_train['event_type']=='reg_input_success'][['date','pv']]
reg_input_uv = data_train[data_train['event_type']=='reg_input_success'][['date','uv']]

#给拆分出的数据集加一列星期的标识
week = ['Thursday','Friday','Saturday','Sunday'] + ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']*28
pageview_pv['week'] = week
pageview_uv['week'] = week
reg_input_pv['week'] = week
reg_input_uv['week'] = week
#目的是划分工作日和休息日
week = [True,True,False,False] + [True,True,True,True,True,False,False]*28
weekend = [False,False,True,True] + [False,False,False,False,False,True,True]*28

pageview_pv_week = pageview_pv[week]
pageview_pv_weekend = pageview_pv[weekend]
pageview_uv_week = pageview_uv[week]
pageview_uv_weekend = pageview_uv[weekend]
reg_input_pv_week = reg_input_pv[week]
reg_input_pv_weekend = reg_input_pv[weekend]
reg_input_uv_week = reg_input_uv[week]
reg_input_uv_weekend = reg_input_uv[weekend]

#定义一个average-moving filter 平滑已知数据
def smooth(y, window_size):
    temp=[]
    for i in range(len(y)):
        if(i<window_size):
            temp.append(sum(y[:i+1])/len(y[:i+1]))
        else:
            temp.append(sum(y[i-window_size+1:i+1])/window_size)
    return temp

#接下来我们去除掉工作日中的节假日
holi = [20181231,20190101,20190204,20190205,20190206,20190207,20190208\
       ,20190405,20190501,20190502,20190503]
pageview_pv_week = pageview_pv_week[~pageview_pv_week['date'].isin(holi)]
pageview_uv_week = pageview_uv_week[~pageview_uv_week['date'].isin(holi+[20190114])]
reg_input_pv_week = reg_input_pv_week[~reg_input_pv_week['date'].isin(holi)]
reg_input_uv_week = reg_input_uv_week[~reg_input_uv_week['date'].isin(holi)]

#加一列时间
pageview_pv_week['num'] = np.arange(131)
pageview_pv_weekend['num'] = np.arange(58)
pageview_uv_week['num'] = np.arange(130)
pageview_uv_weekend['num'] = np.arange(58)
reg_input_pv_week['num'] = np.arange(131)
reg_input_pv_weekend['num'] = np.arange(58)
reg_input_uv_week['num'] = np.arange(131)
reg_input_uv_weekend['num'] = np.arange(58)

window_size = 7
s1 = smooth(pageview_pv_week['pv'],window_size)
s2 = smooth(pageview_pv_weekend['pv'],window_size)
s3 = smooth(pageview_uv_week['uv'],window_size)
s4 = smooth(pageview_uv_weekend['uv'],window_size)
s5 = smooth(reg_input_pv_week['pv'],window_size)
s6 = smooth(reg_input_pv_weekend['pv'],window_size)
s7 = smooth(reg_input_uv_week['uv'],window_size)
s8 = smooth(reg_input_uv_weekend['uv'],window_size)

#在此先计算reginput，计算reg_input的星期波动, pageview将在下面用另一个方法计算
b5 = reg_input_pv_week['pv']/s5
b6 = reg_input_pv_weekend['pv']/s6
b7 = reg_input_uv_week['uv']/s7
b8 = reg_input_uv_weekend['uv']/s8
reg_input_pv_week['new'] = b5
reg_input_pv_weekend['new'] = b6
reg_input_uv_week['new'] = b7
reg_input_uv_weekend['new'] = b8

index5 = reg_input_pv_week[['week','new']].groupby('week').agg(np.mean) 
index6 = reg_input_pv_weekend[['week','new']].groupby('week').agg(np.mean) 
index7 = reg_input_uv_week[['week','new']].groupby('week').agg(np.mean) 
index8 = reg_input_uv_weekend[['week','new']].groupby('week').agg(np.mean) 

#纠正一下星期顺序
print(index5)
print(index6)
print(index7)
print(index8)
index5 = [index5.iloc[1,0],index5.iloc[3,0],index5.iloc[4,0],index5.iloc[2,0],index5.iloc[0,0]]
index6 = [index6.iloc[0,0],index6.iloc[1,0]]
index7 = [index7.iloc[1,0],index7.iloc[3,0],index7.iloc[4,0],index7.iloc[2,0],index7.iloc[0,0]]
index8 = [index8.iloc[0,0],index8.iloc[1,0]]


from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error

#套用ARIMA模型预测reg_input， 因为周末波动较大，我们不采用平滑后的数据，而是采用原数据
temp1=4
temp2=0
temp3=0

model = ARIMA(s5, order=(2,temp2,temp3)).fit()
a_5 = model.predict(131,135)

model = ARIMA(reg_input_pv_weekend['pv'].values, order=(temp1,temp2,temp3)).fit()
a_6 = model.predict(58,59)

model = ARIMA(s7, order=(temp1,temp2,temp3)).fit()
a_7 = model.predict(131,135)

model = ARIMA(reg_input_uv_weekend['uv'].values, order=(temp1,temp2,temp3)).fit()
a_8 = model.predict(58,59)

r_reg_input_pv = np.r_[a_5 * index5 , a_6 * index6]
r_reg_input_uv = np.r_[a_7 * index7 , a_8 * index8]

r_reg_input_uv

#对data_user进行处理，将所有触发时间缺失的行去掉 
data_user = data_user[data_user['xwhen'].isnull()==False]

#定义一个将13位时间戳转化为真实日期的function
def get_time(x):
    if(x>0):
        return time.strftime("%Y-%m-%d", time.localtime(x/1000))
    else:
        return 0
    
#将所有时间戳转换成真实日期
data_user = data_user.sort_values(by='xwhen')
data_user['xwhen'] = data_user['xwhen'].apply(get_time)
data_user['$first_visit_time'] = data_user['$first_visit_time'].apply(get_time)
data_user['$signup_time'] = data_user['$signup_time'].apply(get_time)

#统计每天的user数量
sd=data_user[(data_user['xwhen']>='2019-05-20')&(data_user['xwhen']<'2019-05-27')][['xwhen','$lib']].groupby('xwhen').agg(len).values

#我们通过作图发现工作日真实uv是统计出来的user数量的1.2倍，休息日降为1.1倍
r_pageview_uv = sd[:5]*1.2
r_pageview_uv = np.r_[r_pageview_uv,sd[5:]*1.1].astype(int)

plt.plot(pageview_pv_weekend['pv'].values/pageview_uv_weekend['uv'].values)
print(np.mean(pageview_pv_weekend['pv'].values/pageview_uv_weekend['uv'].values))
print(np.mean((pageview_pv_week['pv'].values[:-1]/pageview_uv_week['uv'].values)[50:]))
#我们可以看到最近的工作日pageview的pv约为uv的3.8倍，所以将uv乘上3.8的得到pageview的pv，休息日约为3.5倍
r_pageview_pv = r_pageview_uv[:5]*3.8
r_pageview_pv = np.r_[r_pageview_pv,r_pageview_uv[5:]*3.5]

#输出
result = pd.DataFrame({'date':[20190520,20190520,20190521,20190521,20190522,20190522,20190523,20190523,20190524,20190524,20190525,20190525,20190526,20190526],\
                     'event_type':['$pageview','reg_input_success','$pageview','reg_input_success','$pageview','reg_input_success','$pageview','reg_input_success','$pageview','reg_input_success','$pageview','reg_input_success','$pageview','reg_input_success'],\
                      'pv':np.r_[r_pageview_pv[0],r_reg_input_pv[0],r_pageview_pv[1],r_reg_input_pv[1],r_pageview_pv[2],r_reg_input_pv[2],r_pageview_pv[3],r_reg_input_pv[3],r_pageview_pv[4],r_reg_input_pv[4],r_pageview_pv[5],r_reg_input_pv[5],r_pageview_pv[6],r_reg_input_pv[6]].astype(int),\
                      'uv':np.r_[r_pageview_uv[0],r_reg_input_uv[0],r_pageview_uv[1],r_reg_input_uv[1],r_pageview_uv[2],r_reg_input_uv[2],r_pageview_uv[3],r_reg_input_uv[3],r_pageview_uv[4],r_reg_input_uv[4],r_pageview_uv[5],r_reg_input_uv[5],r_pageview_uv[6],r_reg_input_uv[6]].astype(int)})
    
    