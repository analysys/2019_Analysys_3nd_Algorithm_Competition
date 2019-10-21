# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:10:23 2019

@author: 41715
"""

import pandas as pd 
import numpy as np
event_detail = pd.read_csv('event_detail.csv')
kpi_train = pd.read_csv('kpi_train.csv')
user = pd.read_csv('user.csv')
a  = event_detail[['time','xwhat','distinct_id']].groupby(['time','xwhat']).count()
b = a.reset_index()
c = b.pivot(index = 'time',columns = 'xwhat',values = 'distinct_id')
c = c.reset_index()
c = c.rename(columns = {'time':'date'})
train_all = kpi_train.merge(c)
temp = event_detail[['time','$is_first_day','$is_login']].rename(columns={'time':'date'})
temp = temp.groupby(['date']).mean().reset_index()
train_all = train_all.merge(temp)
train_all.to_csv('F:/Eguan/第二届是算法大赛/源码/train_all2.csv')

import pandas as pd 
import numpy as np
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import concatenate
from math import sqrt
from matplotlib import pyplot
from sklearn import preprocessing

train = pd.read_csv('datalab/37134/train_all.csv')

def get_data(data,x_cols,y_cols,n_steps,gap):
    df = data.reset_index(inplace=False)
    x = []
    y = []
    for i in range(df.shape[0]-n_steps-gap):
        x_unit = df.loc[i:i+n_steps-1,x_cols].values
        y_unit = df.loc[i+n_steps+gap-1,y_cols].values
        x.append(x_unit)
        y.append(y_unit)
    n_sample = len(x)
    return x,y,n_sample

import math
def df_transform(df,y_cols,df_type):
    x = df.copy()
    x[y_cols] = np.log(x[y_cols])     
    return x

def pred_transform(pred,y_cols,pred_type):
    alpha_set = [0, 0.047, 0.116, 0.039, 0.068, 0.02, 0.058, 0,0]
    return math.exp((1+alpha_set[pred_type])*pred)

def f(df,pred_event_type,gap,n_steps,x_cols,y_cols,change_type):
    train = df[df.event_type == pred_event_type].copy()
    train_change = df_transform(train,y_cols,change_type)
    temp_min = train_change[y_cols].min()
    temp_max = train_change[y_cols].max()
    temp_diff = temp_max - temp_min
    
    train_change = train_change.fillna(0)
    for col in x_cols:
        train_change[col] = (train_change[col] - train_change[col].min()) / (train_change[col].max() - train_change[col].min())
        
    x, y, n_sample = get_data(train_change, x_cols, y_cols, n_steps,gap)
    xx = np.array(x).reshape(n_sample,n_steps,len(x_cols))
    yy = np.array(y).reshape(n_sample,len(y_cols))

    n_train = int(n_sample * 0.9)
    x_train = xx
    y_train = yy
    x_test = xx
    y_test = yy
    
    np.random.seed(0)

    model = Sequential()
    model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    history = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_test, y_test), verbose=2, shuffle=False)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    
    test = xx[-1,:,:]
    test= test.reshape(1,x_train.shape[1],x_train.shape[2])
    pred = model.predict(test[-1:,:,:])[0][0] * temp_diff + temp_min

    return pred_transform(pred,y_cols,change_type)

pv_set = []
uv_set = []
temp_change_type = 1
for gap in range(1,4):
    for pred_event_type in ['$pageview','reg_input_success']:
        if(temp_change_type==1):
            x_cols = ['pv','uv','reg_input_success','reg_submit_click','login','$startup','reg_code_input','$is_first_day','$is_login']
            n_steps = 10
        if(temp_change_type==2):
            x_cols = ['pv','reg_submit_click','login','reg_input_success','$startup','reg_code_input','$is_login']
            n_steps = 10
        if(temp_change_type==3):
            x_cols = ['pv','uv','reg_input_success','reg_submit_click','login','$startup','reg_code_input','$is_first_day','$is_login']
            n_steps = 10
        if(temp_change_type==4): 
            x_cols = ['pv','reg_submit_click','login','reg_input_success','$startup','reg_code_input','$is_login']
            n_step = 10
        if(temp_change_type==5): 
            x_cols = ['pv','uv','reg_input_success','reg_submit_click','login','$startup','reg_code_input','$is_first_day','$is_login']
            n_steps = 10
        if(temp_change_type==6): 
            x_cols = ['pv','reg_submit_click','login','reg_input_success','$startup','reg_code_input','$is_login']
            n_step = 10
        pv_ans = f(train,pred_event_type,gap,n_steps,x_cols,['pv'],temp_change_type)
        pv_set.append(pv_ans)
        if(temp_change_type%2==1): uv_set.append(pv_ans/3.4)
        else: uv_set.append(pv_ans/1.05)
        temp_change_type+=1
        
df_train = train[train.event_type=='$pageview'].copy()
df_train.index = df_train.date
df_train2 = train[train.event_type=='reg_input_success'].copy()
df_train2.index = df_train.date

beta = 0.8
pv_mix = beta * df_train.loc[20190516,'pv'] + (1-beta)* df_train.loc[20190509,'pv']
pv_set.append(pv_mix)
pv_set.append(pv_set[-2])
uv_set.append(pv_mix / 3.5)
uv_set.append(uv_set[-2])
pv_mix = beta * df_train.loc[20190517,'pv'] + (1-beta)* df_train.loc[20190510,'pv']
pv_set.append(pv_mix)
pv_set.append(pv_set[-2])
uv_set.append(pv_mix / 3.5)
uv_set.append(uv_set[-2])

beta2 = 0.5
pv_mix = beta2 * df_train.loc[20190518,'pv'] + (1-beta2)* df_train.loc[20190511,'pv']
pv_set.append(pv_mix)
uv_set.append(pv_mix / 3)
pv_mix = beta2 * df_train2.loc[20190518,'pv'] + (1-beta2)* df_train2.loc[20190511,'pv']
pv_set.append(pv_mix)
uv_set.append(pv_mix)

beta3 = 1.4
pv_mix = beta3 * df_train.loc[20190519,'pv'] + (1-beta3)* df_train.loc[20190512,'pv']
pv_set.append(pv_mix)
uv_set.append(df_train.loc[20190519,'uv'])

beta4 = 0.5
pv_mix = beta4 * df_train2.loc[20190519,'pv'] + (1-beta4)* df_train2.loc[20190512,'pv']
pv_set.append(pv_mix)
uv_set.append(pv_mix)

date = ['20190520','20190520','20190521','20190521','20190522','20190522','20190523','20190523','20190524','20190524','20190525','20190525','20190526','20190526']
event_type = ['$pageview','reg_input_success'] *7
sub = pd.DataFrame({'date':date,'event_type':event_type,'pv':pv_set,'uv':uv_set})

for col in ['pv','uv']:
    sub[col] = sub[col].astype('int')
