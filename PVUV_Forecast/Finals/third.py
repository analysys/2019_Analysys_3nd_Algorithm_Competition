
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
train = pd.read_csv('event_detail.csv')
for i in range(1,6):
    temp = pd.read_csv(f'event_detail_{i}.csv')
    train = pd.concat([train,temp],axis = 0)
kpi = pd.read_csv('kpi_train.csv')
kpi_new = pd.read_csv('kpi_train_new.csv')
kpi_train = pd.concat([kpi,kpi_new],axis = 0)
event_detail = train
a  = event_detail[['time','xwhat','distinct_id']].groupby(['time','xwhat']).count()
b = a.reset_index()
c = b.pivot(index = 'time',columns = 'xwhat',values = 'distinct_id')
c = c.reset_index()
c = c.rename(columns = {'time':'date'})
train_all = kpi_train.merge(c)
temp = event_detail[['time','$is_first_day','$is_login']].rename(columns={'time':'date'})
temp = temp.groupby(['date']).mean().reset_index()
train_all = train_all.merge(temp)
train_all.to_csv('train_all_new.csv')


# In[22]:


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

train = pd.read_csv('train_all_new.csv')



train = train.drop_duplicates(subset=['date','event_type'], keep='first', inplace=False)



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
    alpha_set = [0, 0.037, 0.063, 0.033, 0.119, 0.039, 0.156, 0, 0, 0]
    return math.exp((1+alpha_set[pred_type])*pred)




def f(df,pred_event_type,gap,n_steps,x_cols,y_cols,change_type):
    train = df[df.event_type == pred_event_type].copy()
    train_change = df_transform(train,y_cols,change_type)
    temp_min = train_change[y_cols].min()
    temp_max = train_change[y_cols].max()
    temp_diff = temp_max - temp_min
    
    train_change = train_change.fillna(0)
    for col in x_cols:
        train_change[col] = (train_change[col] - train_change[col].min()) / (train_change[col].max() - train_change[col].min()) #标准化
        
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
            n_steps = 15
        if(temp_change_type==2):
            x_cols = ['pv','reg_submit_click','login','reg_input_success','$startup','reg_code_input','$is_login']
            n_steps = 15
        if(temp_change_type==3):
            x_cols = ['pv','uv','reg_input_success','reg_submit_click','login','$startup','reg_code_input','$is_first_day','$is_login']
            n_steps = 20
        if(temp_change_type==4): 
            x_cols = ['pv','reg_submit_click','login','reg_input_success','$startup','reg_code_input','$is_login']
            n_step = 20
        if(temp_change_type==5): 
            x_cols = ['pv','uv','reg_input_success','reg_submit_click','login','$startup','reg_code_input','$is_first_day','$is_login']
            n_steps = 25
        if(temp_change_type==6): 
            x_cols = ['pv','reg_submit_click','login','reg_input_success','$startup','reg_code_input','$is_login']
            n_step = 25
        pv_ans = f(train,pred_event_type,gap,n_steps,x_cols,['pv'],temp_change_type)
        pv_set.append(pv_ans)
        if(temp_change_type==1): uv_set.append(pv_ans/3.4)
        elif(temp_change_type%2==1): uv_set.append(pv_ans/3.1)
        else: uv_set.append(pv_ans/1)
        temp_change_type+=1



df_train = train[train.event_type=='$pageview'].copy()
df_train.index = df_train.date
df_train2 = train[train.event_type=='reg_input_success'].copy()
df_train2.index = df_train.date

beta = 0.5
pv_mix = beta * df_train.loc[20190822,'pv'] + (1-beta)* df_train.loc[20190815,'pv']
pv_set.append(pv_mix)
uv_set.append(pv_mix / 3.3)
pv_mix = beta * df_train2.loc[20190822,'uv'] + (1-beta)* df_train2.loc[20190815,'uv']
pv_set.append(pv_mix)
uv_set.append(pv_mix)

beta = 0.5
pv_mix = beta * df_train.loc[20190823,'pv'] + (1-beta)* df_train.loc[20190816,'pv']
pv_set.append(pv_mix)
uv_set.append(pv_mix / 3)
beta = 0.9
pv_mix = beta * df_train2.loc[20190823,'uv'] + (1-beta)* df_train2.loc[20190816,'uv']
pv_set.append(pv_mix)
uv_set.append(pv_mix)


beta2 = 0.5
pv_mix = beta2 * df_train.loc[20190824,'pv'] + (1-beta2)* df_train.loc[20190817,'pv']
pv_set.append(pv_mix)
uv_set.append(pv_mix / 2.5)
pv_mix = beta2 * df_train2.loc[20190824,'uv'] + (1-beta2)* df_train2.loc[20190817,'uv']
pv_set.append(pv_mix)
uv_set.append(pv_mix)

beta2 = 0.5
pv_mix = beta2 * df_train.loc[20190825,'pv'] + (1-beta2)* df_train.loc[20190818,'pv']
pv_set.append(pv_mix)
uv_set.append(pv_mix / 2.5)
beta2 = 0.01
pv_mix = beta2 * df_train2.loc[20190825,'uv'] + (1-beta2)* df_train2.loc[20190818,'uv']
pv_set.append(pv_mix)
uv_set.append(pv_mix)



date = ['20190826','20190826','20190827','20190827','20190828','20190828','20190829','20190829','20190830','20190830','20190831','20190831','20190901','20190901']
event_type = ['$pageview','reg_input_success'] *7
sub = pd.DataFrame({'date':date,'event_type':event_type,'pv':pv_set,'uv':uv_set})



for col in ['pv','uv']:
    sub[col] = sub[col].astype('int')


sub

