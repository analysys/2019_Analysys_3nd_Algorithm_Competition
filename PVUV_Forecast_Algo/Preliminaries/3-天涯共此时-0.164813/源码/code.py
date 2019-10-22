import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
from datetime import timedelta,datetime
from fbprophet import Prophet
from sklearn.preprocessing import MinMaxScaler


data_dir='./'
kpi_train=pd.read_csv(data_dir+'kpi_train.csv')
kpi_train.date=pd.to_datetime(kpi_train.date, format='%Y%m%d')
page_uv_train=kpi_train[kpi_train.event_type=='$pageview'][['date','uv']].sort_values('date').reset_index(drop=1==1).rename(columns={'date':'ds','uv':'y'})
reg_uv_train=kpi_train[kpi_train.event_type=='reg_input_success'][['date','uv']].sort_values('date').reset_index(drop=1==1).rename(columns={'date':'ds','uv':'y'})
page_pv_train=kpi_train[kpi_train.event_type=='$pageview'][['date','pv']].sort_values('date').reset_index(drop=1==1).rename(columns={'date':'ds','pv':'y'})
reg_pv_train=kpi_train[kpi_train.event_type=='reg_input_success'][['date','pv']].sort_values('date').reset_index(drop=1==1).rename(columns={'date':'ds','pv':'y'})


def Grey_model(x,n): #灰色预测
    x1 = x.cumsum() #累计求次数
    z1 = (x1[:len(x1) - 1] + x1[1:])/2.0 
    z1 = z1.reshape((len(z1),1))  
    B = np.append(-z1,np.ones_like(z1),axis=1)  
    Y = x[1:].reshape((len(x) - 1,1))
    
    [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)#计算参数  
    result = (x[0]-b/a)*np.exp(-a*(n-1))-(x[0]-b/a)*np.exp(-a*(n-2))  
    S1_2 = x.var() 
    e = list() 
    for index in range(1,x.shape[0]+1):
        predict = (x[0]-b/a)*np.exp(-a*(index-1))-(x[0]-b/a)*np.exp(-a*(index-2))
        e.append(x[index-1]-predict)
    S2_2 = np.array(e).var() 
    C = S2_2/S1_2 
    if C<=0.35:
        assess = 'C <= 0.35，The accuracy level of the model is very good'
    elif C<=0.5:
        assess = 'C <= 0.5，The accuracy level of the model is good'
    elif C<=0.65:
        assess = 'C <= 0.65，The accuracy level of the model is bad'
    else:
        assess = 'C <= 0.65，The accuracy level of the model is very bad'
     
    predict = list()
    for index in range(x.shape[0]+1,x.shape[0]+n+1):
        predict.append((x[0]-b/a)*np.exp(-a*(index-1))-(x[0]-b/a)*np.exp(-a*(index-2)))
    predict = np.array(predict)
    return {
            'a':{'value':a},
            'b':{'value':b},
            'C':{'value':C,'desc':assess},
            'predict':{'value':predict},
            }

page_uv_Gpre=np.round(Grey_model(page_uv_train.iloc[102:,:].y.values,  7)['predict']['value'])
page_pv_Gpre=np.round(Grey_model(page_pv_train.iloc[102:,:].y.values,  7)['predict']['value']) 
reg_uv_Gpre=np.round(Grey_model(reg_uv_train.iloc[102:,:].y.values,  7)['predict']['value'])
reg_pv_Gpre=np.round(Grey_model(reg_pv_train.iloc[102:,:].y.values,  7)['predict']['value'])



m1=Prophet(changepoint_prior_scale=1.4)  
# page_uv_train_qyc=page_uv_train.copy()
# page_uv_train_qyc.iloc[88:102,1]=None #2019-1-28至2019-2-10这两周异常值?
# page_uv_train_lg=page_uv_train.copy()
# page_uv_train_lg.y=page_uv_train_lg.y.apply(lambda x : np.log(x)) 
m1.fit(page_uv_train.iloc[102:,:]) #从2019-2-11开始预测
page_uv_future=m1.make_future_dataframe(periods=7)
page_uv_pre=m1.predict(page_uv_future)


# m2=Prophet(changepoint_prior_scale=0.75)  
# m2.fit(page_pv_train.iloc[102:,:])
# page_pv_future=m2.make_future_dataframe(periods=7)
# page_pv_pre=m2.predict(page_pv_future)


m3=Prophet(changepoint_prior_scale=0.1) 
# reg_uv_train_qyc=reg_uv_train.copy()
# reg_uv_train_qyc.iloc[198:200,1]=None  
# m3.fit(reg_uv_train_qyc.iloc[102:,:])
m3.fit(reg_uv_train.iloc[102:,:])
reg_uv_future=m3.make_future_dataframe(periods=7)
reg_uv_pre=m3.predict(reg_uv_future)


m4=Prophet(changepoint_prior_scale=3) 
reg_chazhi=reg_pv_train.copy()
reg_chazhi.y=reg_pv_train.y-reg_uv_train.y
reg_chazhi.iloc[198:200,1]=None   
m4.fit(reg_chazhi.iloc[102:,:])
reg_chazhi_future=m4.make_future_dataframe(periods=7)
reg_chazhi_pre=m4.predict(reg_chazhi_future)

 


p=page_uv_pre.tail(7)[['ds','yhat']].rename(columns={'ds':'date', 'yhat':'uv'})
p.loc[p.date.dt.weekday==2, 'uv']/=1.17  
p['pv']=p.uv #
p.loc[p.date.dt.weekday<5, 'pv']*=3.75
p.loc[p.date.dt.weekday>=5, 'pv']*=3
p['event_type']='$pageview'
 
    
r=reg_uv_pre.tail(7)[['ds','yhat']].rename(columns={'ds':'date', 'yhat':'uv'})
r.loc[r.date.dt.weekday==0, 'uv']*=1.17 
avg=(r.loc[r.date.dt.weekday>=5, 'uv']).sum()/2
r.loc[r.date.dt.weekday==5, 'uv']=avg
r.loc[r.date.dt.weekday==6, 'uv']=avg
r.loc[r.date.dt.weekday==2, 'uv']/=1.1
mm=MinMaxScaler(feature_range=(0,4))
r['pv']=r.uv+pd.DataFrame(mm.fit_transform(reg_chazhi_pre.tail(7).yhat.values.reshape(-1, 1)) ,index=list(range(98,105))).iloc[:,0]
r['event_type']='reg_input_success'   

sub_temp=pd.concat([p,r], ignore_index=True).sort_values('date').reset_index(drop=1==1)
sub_temp.date=sub_temp.date.astype(str).apply(lambda x : x.replace('-',''))
sub_temp[['uv','pv']]=sub_temp[['uv','pv']].astype(int)

 



sub_temp[['date','event_type', 'pv', 'uv']].to_csv('submission.csv',index=False) 