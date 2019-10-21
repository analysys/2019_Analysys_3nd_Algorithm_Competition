import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
from datetime import timedelta,datetime
from fbprophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import gc
from fbprophet.plot import plot_plotly
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode()
%matplotlib inline
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',600)

 
def num_range(df):
    sll = df.select_dtypes(include=['int64','int32','int16','int8','float32'])
    for cols in sll.columns:
        print(cols, sll[cols].min(), sll[cols].max())      
def num_downcast(df):
    int_cols = df.select_dtypes(include=['int']).columns  
    df[int_cols] = df[int_cols].apply(pd.to_numeric,downcast='signed')
    float_cols = df.select_dtypes(include=['float']).columns
    df[float_cols] = df[float_cols].apply(pd.to_numeric,downcast='float')
    
    
path='./'
kpi_train=pd.read_csv(path+'kpi_train_new.csv')
ed0=pd.read_csv(path+'event_detail.csv')
ed1=pd.read_csv(path+'event_detail_1.csv')
ed2=pd.read_csv(path+'event_detail_2.csv')
ed3=pd.read_csv(path+'event_detail_3.csv')
ed4=pd.read_csv(path+'event_detail_4.csv')
ed5=pd.read_csv(path+'event_detail_5.csv')
ed=pd.concat([ed0,ed1,ed2,ed3,ed4,ed5], ignore_index=True)
del ed0,ed1,ed2,ed3,ed4,ed5
gc.collect()    
 
kpi_train.date=pd.to_datetime(kpi_train.date, format='%Y%m%d')
page_uv_train=kpi_train[kpi_train.event_type=='$pageview'][['date','uv']].sort_values('date').reset_index(drop=1==1).rename(columns={'date':'ds','uv':'y'})
reg_uv_train=kpi_train[kpi_train.event_type=='reg_input_success'][['date','uv']].sort_values('date').reset_index(drop=1==1).rename(columns={'date':'ds','uv':'y'})
page_pv_train=kpi_train[kpi_train.event_type=='$pageview'][['date','pv']].sort_values('date').reset_index(drop=1==1).rename(columns={'date':'ds','pv':'y'})
reg_pv_train=kpi_train[kpi_train.event_type=='reg_input_success'][['date','pv']].sort_values('date').reset_index(drop=1==1).rename(columns={'date':'ds','pv':'y'})

#预测的是8月底的，临近收假，中间隔有暑假这一段特殊日期，而且观察时序图发现，近三周的数据变化较为平稳，所以我们便采用7.15之后的数据来预测
m1=Prophet(changepoint_prior_scale=1.4)  
page_uv_train_qyc=page_uv_train.copy()
page_uv_train_qyc.iloc[272,1]=None #去掉7-31的异常值?
m1.fit(page_uv_train_qyc.iloc[256:,:]) #221 
page_uv_future=m1.make_future_dataframe(periods=7)
page_uv_pre=m1.predict(page_uv_future)

m2=Prophet(changepoint_prior_scale=0.1)  
reg_uv_train_qyc=reg_uv_train.copy()
reg_uv_train_qyc.iloc[256:277,1]=None #去掉7-15到8-4的异常值?  
m2.fit(reg_uv_train_qyc.iloc[256:,:])
reg_uv_future=m2.make_future_dataframe(periods=7)
reg_uv_pre=m2.predict(reg_uv_future) 
    
p=page_uv_pre.tail(7)[['ds','yhat']].rename(columns={'ds':'date', 'yhat':'uv'})
p['pv']=p.uv
p.loc[p.date.dt.weekday<5, 'pv']*=3 #近2-3周的pv/uv
p.loc[p.date.dt.weekday>=5, 'pv']*=2.5 #近2-3周的pv/uv
p['event_type']='$pageview'
 
r=reg_uv_pre.tail(7)[['ds','yhat']].rename(columns={'ds':'date', 'yhat':'uv'})
r['pv']=r.uv  
r['event_type']='reg_input_success'   

sub_temp=pd.concat([p,r], ignore_index=True).sort_values('date').reset_index(drop=1==1)
sub_temp.date=sub_temp.date.astype(str).apply(lambda x : x.replace('-',''))
sub_temp[['uv','pv']]=sub_temp[['uv','pv']].astype(int)

sub_temp[['date','event_type', 'pv', 'uv']].to_csv('submission.csv',index=False) 
    
    
    
    