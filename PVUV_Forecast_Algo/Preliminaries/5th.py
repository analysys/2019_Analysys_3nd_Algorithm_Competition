
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import warnings
import time
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import lightgbm as lgb
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm, tqdm_notebook
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import math
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
# from gensim.models import Word2Vec
import json
import gc
import re
warnings.filterwarnings('ignore')


import lightgbm as lgb

def eval_score(preds, train_data):
    labels = train_data.get_label()
    df = pd.DataFrame()
    df['y_true'] = labels
    df['y_pred'] = preds
    df['sc'] = ((df['y_pred'] - df['y_true'] )/df['y_true']) ** 2
    score = np.sqrt(df.sc.mean())
    
    return 'npmse', score, False

def model_npmse_predict(params_in,train,valid,test,pred1):
    'predict_in'
#     params_in ={
#         'learning_rate': 0.01,
#         'boosting_type': 'gbdt',
#         'objective': 'regression',
#         'metric': 'mae',
#         'num_leaves': 400,
#         'feature_fraction': 0.7,
#         'bagging_fraction': 0.9,
#         'bagging_freq': 10,
#         'seed': 666,
#         'bagging_seed': 10001,
#         'feature_fraction_seed': 7,
#         'min_data_in_leaf': 20,
#         'reg_sqrt': True,
#         'nthread': 8,
#         'verbose': -1,
#     }
    
    
    
# ,categorical_feature=cateCols
# ,categorical_feature=cateCols
    dtrain = lgb.Dataset(train[feats],free_raw_data =False,  label=train[target])
    dvalid = lgb.Dataset(valid[feats], free_raw_data =False, label=valid[target])
    gbm = lgb.train(
  	 params=params_in,
  	 train_set=dtrain,
  	 num_boost_round=500,
  	 valid_sets=[dvalid],
         early_stopping_rounds=30,
    	 feval = eval_score,
    	 verbose_eval=10
	)
    
    pred = gbm.predict(test[pred1])
    
    return pred,gbm

def score(preds, labels):
    df = pd.DataFrame()
    df['y_true'] = labels
    df['y_pred'] = preds
    df['sc'] = ((df['y_pred'] - df['y_true'] )/df['y_true']) ** 2
    sc = np.sqrt(df.sc.mean())
    
    return sc
    


def model_logged_npmse_predict(params_in,train,valid,test,pred1):
    'predict_in'
#     params_in ={
#         'learning_rate': 0.01,
#         'boosting_type': 'gbdt',
#         'objective': 'regression',
#         'metric': 'mae',
#         'num_leaves': 400,
#         'feature_fraction': 0.7,
#         'bagging_fraction': 0.9,
#         'bagging_freq': 10,
#         'seed': 666,
#         'bagging_seed': 10001,
#         'feature_fraction_seed': 7,
#         'min_data_in_leaf': 20,
#         'reg_sqrt': True,
#         'nthread': 8,
#         'verbose': -1,
#     }
    
    
    
# ,categorical_feature=cateCols
# ,categorical_feature=cateCols
    dtrain = lgb.Dataset(train[feats],free_raw_data =False,  label=np.log1p(train[target]))
    dvalid = lgb.Dataset(valid[feats], free_raw_data =False, label=np.log1p(valid[target]))
    gbm = lgb.train(
  	 params=params_in,
  	 train_set=dtrain,
  	 num_boost_round=1000,
  	 valid_sets=[dvalid],
         early_stopping_rounds=30,
    	 feval = eval_score,
    	 verbose_eval=10
	)
    
    pred = gbm.predict(test[pred1])
    
    return pred,gbm



from sklearn.metrics import mean_absolute_error as mae

def correction(preds,labels):
    weights = np.arange(0.5, 1.02, 0.005)
    errors = []
    for w in weights:
        
        sc = score(preds*w, labels)
        errors.append(sc)
    idx = errors.index(min(errors))
    print('Best weight is {}, mae Score is {:.4f}'.format(weights[idx], min(errors)))
    return weights[idx]
import os

import matplotlib.pyplot as plt
import seaborn as sns
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by = "importance", ascending = False)[:50
                                                                                                                                        ].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    # print(feature_importance_df_[feature_importance_df_.feature.map(lambda x:'IS' in x)])
    # print(best_features.feature.tolist())
    plt.figure(figsize = (15, 10))
    sns.barplot(x = "importance", y = "feature", data = best_features.sort_values(by = "importance", ascending = False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
#     plt.savefig('../cache/lgbm_importances.png')
def get_Impdf(lgb_model,feats,fileIndex ):
    feature_importance_df = pd.DataFrame()
    feature_importance_df['feature'] = feats
    feature_importance_df['importance'] = lgb_model.feature_importance()
    # imp_df[imp_df.fea.map(lambda x:'ip_first' not in x)==True].sort_values(by=['imp'], ascending=False).tail(90)
    import datetime
    td = datetime.datetime.today()
    tdstr = str(td)[:10]
    # save_imp_csv = '../cache/FeaImp_{}_{}.csv'.format( tdstr,fileIndex )
    display_importances(feature_importance_df)


def check_qulify(df,dateCol,rollBy):
	gp = df.groupby([ dateCol,rollBy ] )[ dateCol ].agg({'count':'count'}).reset_index()
	if gp['count'].nunique()!=1:
		raise TypeError(' rollBy ERROR!!! combination of dateCol and rollBy must be unique')
	del gp



def My_rolling( df , rollBy,rollOn,shif_win,rolls_win,dateCol ):
	if isinstance(rolls_win,list):
		pass
	else:
		raise TypeError(' rolls_win must be a list  ')
	if min(rolls_win)==1:
		raise TypeError(' min value of rolls_win ==1  ')
	check_qulify(df,dateCol,rollBy)

# 	df = df.sort_values(dateCol).reset_index(drop=True)
	for day in rolls_win:
		tmp_shifts = []
		for rol in range(1,day+1):
			thshif = shif_win + rol-1
			tm = 'shif_{}_num'.format( thshif )
			df[tm] = df.groupby([rollBy])[rollOn].shift( thshif )
			tmp_shifts.append( tm )
		if day == 2:
			feat = 'base_{}_rolling_{}_mean_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].mean(axis=1)
		else:
			feat = 'base_{}_rolling_{}_mean_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].mean(axis=1)
			feat = 'base_{}_rolling_{}_max_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].max(axis=1)
			feat = 'base_{}_rolling_{}_min_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].min(axis=1)
			feat = 'base_{}_rolling_{}_median_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].median(axis=1)
			feat = 'base_{}_rolling_{}_skew_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].skew(axis=1)
		df.drop(tmp_shifts,axis=1,inplace=True)
	return df


# In[2]:


data = pd.read_csv('kpi_train.csv')
sub = pd.read_csv('kpi_test.csv')
test = sub.copy()
test['pv'] = np.nan
test['uv'] = np.nan
data = data.append(test).reset_index(drop=True)


# In[3]:


def past_m_summary(df,shif_win,rollBy,rollOn,dayRnge,dateCol):
    tmp_shifts = []
    use_cols = [rollBy,rollOn,dateCol]
    result = pd.DataFrame()
    minDy = dayRnge[0]
    maxDy = dayRnge[1]
    for day in range(minDy,maxDy+1):
        _df = df[df[dateCol]==day][use_cols]
        if day-shif_win< 2:
            pass
        elif day-shif_win== 2:
            dfs = df[df[dateCol] <= day-shif_win]
            gp = dfs.groupby([rollBy])[rollOn].mean()
            _df['Past_{}_Summ_{}_{}'.format( shif_win,rollOn,'mean' )] = _df[rollBy].map(gp)
        else:
            dfs = df[df[dateCol] <= day-shif_win]
            gp = dfs.groupby([rollBy])[rollOn].mean()
            _df['Past_{}_Summ_{}_{}'.format( shif_win,rollOn,'mean' )] = _df[rollBy].map(gp)
            gp = dfs.groupby([rollBy])[rollOn].median()
            _df['Past_{}_Summ_{}_{}'.format( shif_win,rollOn,'median' )] = _df[rollBy].map(gp)
            gp = dfs.groupby([rollBy])[rollOn].skew()
            _df['Past_{}_Summ_{}_{}'.format( shif_win,rollOn,'skew' )] = _df[rollBy].map(gp)
        result = pd.concat([result,_df])
            
    del result[rollOn]
    del _df
    df = df.merge(result,on=[rollBy,dateCol],how='left' )
    del result
    return df
    


# In[4]:


target = 'pv'
type_ = 'reg_input_success'
# reg_input_success
# '$pageview'
# 20190304
usedata = data[data.date>=20190304].reset_index(drop=True)
reg_input_data = usedata[usedata.event_type == type_ ][['date',target]]
reg_input_data['dow'] = pd.to_datetime(reg_input_data['date'].map(str)).dt.dayofweek
reg_input_data['weekOfYear'] = pd.to_datetime(reg_input_data['date'].map(str)).dt.weekofyear
reg_input_data = reg_input_data[reg_input_data.weekOfYear>=10]
reg_input_data['weekOfYear'] = reg_input_data['weekOfYear']-9


df = reg_input_data.copy()
df = df.sort_values('date').reset_index(drop=True)
df['dow_dum_'] = df['dow'].copy()
df = pd.get_dummies(df,columns=['dow_dum_'])

sorted_unique_date = sorted(df.date.unique().tolist())
date_order_dict = dict( zip( sorted_unique_date, range(1,len(sorted_unique_date)+1 ) ) )
df['dayIndex'] = df['date'].map(date_order_dict)
df = df.sort_values('dayIndex').reset_index(drop = True)
df['none'] = 1
dayRnge = [df.dayIndex.min(), df.dayIndex.max()]
for shi in [7,10,14]:
    df = past_m_summary(df,shi,'none',target,dayRnge,'dayIndex')


# df['is_monday'] = 0
# df.loc[df['dow']==0,'is_monday'] = 1

rollBy = 'dow'
rollOn = target
shif_wins = [1,2,3]
rolls_win = [2,3]
for shif_win in shif_wins:
    df = My_rolling( df , rollBy,rollOn,shif_win,rolls_win,'weekOfYear' )
df['weekday_df'] = df.groupby(['dow'])[target].diff()
df['weekday_df'] = df.groupby(['dow'])['weekday_df'].shift(1)
base_day = df.weekOfYear.max()
df = df.fillna( df.mean())
# df.loc[df.dow == 0,'dow']=np.nan


# train = train[train.date<20190503].reset_index(drop=True)
# train


# In[5]:


# df.loc[df.dow == 0,'dow']=np.nan
test = df.loc[ df.weekOfYear==base_day ].reset_index(drop=True)
valid = df.loc[ df.weekOfYear.isin([base_day-2,base_day-1]) ].reset_index(drop=True)

train = df.loc[ df.weekOfYear<base_day-2 ].reset_index(drop=True)
print(train.date.max())


# In[6]:


feats = [c for c in df.columns  if c not in ['date','weekOfYear','pv']]


# In[7]:



params_in = {'num_leaves': 16,
        #   'min_child_weight': 0.044,
          'feature_fraction': 0.8,
          'bagging_fraction': 0.8,
                 'bagging_freq': 10,
          'min_data_in_leaf': 14,
          'objective': 'regression',
          'max_depth': -1,
          'learning_rate': 0.07,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          'metric': 'rmse',
          'num_threads': 10,
          "verbosity": -1,
          'reg_alpha': 0.038,
          'reg_lambda': 0.5,
          'random_state': 88
         }



# In[8]:


import pandas as pd
import numpy as np
#Import packages for preprocessing (data cleaning)
from sklearn.preprocessing import RobustScaler, LabelEncoder
#Import packages for data visualisation
import matplotlib.pyplot as plt
#Import packages for model testing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
#Import packages for modeling
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline
data_to_train = train.append(valid).reset_index(drop=True)
labels_to_use = data_to_train[target]
data_to_train = data_to_train[feats]

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0007, random_state=1))
ridge = make_pipeline(RobustScaler(), Ridge(alpha =0.0007, random_state=42))
def evaluation(model):
    result= np.sqrt(-cross_val_score(model, data_to_train, labels_to_use, cv = 5, scoring = 'neg_mean_squared_error'))
    return(result)
ls_score = evaluation(lasso)
print("Lasso score: {:.5f}\n".format(ls_score.mean()))


# In[9]:


feats=['dow',
 'dow_dum__4',
 'dow_dum__3',
 'base_1_rolling_3_mean_pv_OF_dow',
 'base_2_rolling_3_min_pv_OF_dow',
 'base_1_rolling_3_skew_pv_OF_dow',
 'Past_7_Summ_pv_skew',
 'base_3_rolling_3_skew_pv_OF_dow',
 'base_2_rolling_3_skew_pv_OF_dow',
 'none']


# In[10]:


ridge.fit(train[feats ], np.log1p(train[target]))
test_pred_1 = ridge.predict(test[ feats])
valid_pred  = ridge.predict(valid[feats])
weight = correction(valid_pred,np.log1p(valid[target]))
pre2 = np.expm1(test_pred_1*weight)


# In[11]:


feats=['dow',
 'dow_dum__4',
 'dow_dum__3',
 'base_1_rolling_3_mean_pv_OF_dow',
 'base_2_rolling_3_min_pv_OF_dow',
 'base_3_rolling_3_skew_pv_OF_dow',
 'none']


# In[12]:


lasso.fit(train[feats ], np.log1p(train[target]))
test_pred_1 = lasso.predict(test[ feats])
valid_pred  = lasso.predict(valid[feats])
weight = correction(valid_pred,np.log1p(valid[target]))
pre3 = np.expm1(test_pred_1*weight)


# In[13]:



feats=['dow',
 'weekOfYear',
 'dow_dum__0',
 'dow_dum__1',
 'dow_dum__2',
 'dow_dum__3',
 'dow_dum__4',
 'dow_dum__5',
 'dow_dum__6',
 'dayIndex']

pv_pred,gbm =  model_logged_npmse_predict(params_in,train,valid,test,feats)

test_pred_1 = gbm.predict(test[ feats])
valid_pred  = gbm.predict(valid[feats])
weight = correction(valid_pred,np.log1p(valid[target]))
pv_pre = np.expm1(test_pred_1*weight)


# In[14]:


r_n = (10*pre2 + (np.array(pv_pre)*(40))+(np.array(pre3)*(50)))/100
reg_pv=r_n


# In[15]:


reg_uv=reg_pv


# In[16]:


import pandas as pd
import numpy as np
import os
import warnings
import time
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import lightgbm as lgb
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm, tqdm_notebook
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import math
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
# from gensim.models import Word2Vec
import json
import gc
import re
warnings.filterwarnings('ignore')


import lightgbm as lgb

def eval_score(preds, train_data):
    labels = train_data.get_label()
    df = pd.DataFrame()
    df['y_true'] = labels
    df['y_pred'] = preds
    df['sc'] = ((df['y_pred'] - df['y_true'] )/df['y_true']) ** 2
    score = np.sqrt(df.sc.mean())
    
    return 'npmse', score, False



def model_npmse_predict(params_in,train,valid,test,pred1):
    'predict_in'
#     params_in ={
#         'learning_rate': 0.01,
#         'boosting_type': 'gbdt',
#         'objective': 'regression',
#         'metric': 'mae',
#         'num_leaves': 400,
#         'feature_fraction': 0.7,
#         'bagging_fraction': 0.9,
#         'bagging_freq': 10,
#         'seed': 666,
#         'bagging_seed': 10001,
#         'feature_fraction_seed': 7,
#         'min_data_in_leaf': 20,
#         'reg_sqrt': True,
#         'nthread': 8,
#         'verbose': -1,
#     }
    
    
    
# ,categorical_feature=cateCols
# ,categorical_feature=cateCols
    dtrain = lgb.Dataset(train[feats],free_raw_data =False,  label=train[target])
    dvalid = lgb.Dataset(valid[feats], free_raw_data =False, label=valid[target])
    gbm = lgb.train(
  	 params=params_in,
  	 train_set=dtrain,
  	 num_boost_round=500,
  	 valid_sets=[dvalid],
         early_stopping_rounds=30,
    	 feval = eval_score,
    	 verbose_eval=10
	)
    
    pred = gbm.predict(test[pred1])
    
    return pred,gbm


    


def model_logged_npmse_predict(params_in,train,valid,test,pred1):
    'predict_in'
#     params_in ={
#         'learning_rate': 0.01,
#         'boosting_type': 'gbdt',
#         'objective': 'regression',
#         'metric': 'mae',
#         'num_leaves': 400,
#         'feature_fraction': 0.7,
#         'bagging_fraction': 0.9,
#         'bagging_freq': 10,
#         'seed': 666,
#         'bagging_seed': 10001,
#         'feature_fraction_seed': 7,
#         'min_data_in_leaf': 20,
#         'reg_sqrt': True,
#         'nthread': 8,
#         'verbose': -1,
#     }
    
    
    
# ,categorical_feature=cateCols
# ,categorical_feature=cateCols
    dtrain = lgb.Dataset(train[feats],free_raw_data =False,  label=np.log1p(train[target]))
    dvalid = lgb.Dataset(valid[feats], free_raw_data =False, label=np.log1p(valid[target]))
    gbm = lgb.train(
  	 params=params_in,
  	 train_set=dtrain,
  	 num_boost_round=1000,
  	 valid_sets=[dvalid],
         early_stopping_rounds=30,
    	 feval = eval_score,
    	 verbose_eval=10
	)
    
    pred = gbm.predict(test[pred1])
    
    return pred,gbm



from sklearn.metrics import mean_absolute_error as mae

def correction(preds,labels):
    weights = np.arange(0.5, 1.02, 0.005)
    errors = []
    for w in weights:
        
        sc = score(preds*w, labels)
        errors.append(sc)
    idx = errors.index(min(errors))
    print('Best weight is {}, mae Score is {:.4f}'.format(weights[idx], min(errors)))
    return weights[idx]
import os

import matplotlib.pyplot as plt
import seaborn as sns
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by = "importance", ascending = False)[:50
                                                                                                                                        ].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    # print(feature_importance_df_[feature_importance_df_.feature.map(lambda x:'IS' in x)])
    # print(best_features.feature.tolist())
    plt.figure(figsize = (15, 10))
    sns.barplot(x = "importance", y = "feature", data = best_features.sort_values(by = "importance", ascending = False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
#     plt.savefig('../cache/lgbm_importances.png')
def get_Impdf(lgb_model,feats,fileIndex ):
    feature_importance_df = pd.DataFrame()
    feature_importance_df['feature'] = feats
    feature_importance_df['importance'] = lgb_model.feature_importance()
    # imp_df[imp_df.fea.map(lambda x:'ip_first' not in x)==True].sort_values(by=['imp'], ascending=False).tail(90)
    import datetime
    td = datetime.datetime.today()
    tdstr = str(td)[:10]
    # save_imp_csv = '../cache/FeaImp_{}_{}.csv'.format( tdstr,fileIndex )
    display_importances(feature_importance_df)


def check_qulify(df,dateCol,rollBy):
	gp = df.groupby([ dateCol,rollBy ] )[ dateCol ].agg({'count':'count'}).reset_index()
	if gp['count'].nunique()!=1:
		raise TypeError(' rollBy ERROR!!! combination of dateCol and rollBy must be unique')
	del gp



def My_rolling( df , rollBy,rollOn,shif_win,rolls_win,dateCol ):
	if isinstance(rolls_win,list):
		pass
	else:
		raise TypeError(' rolls_win must be a list  ')
	if min(rolls_win)==1:
		raise TypeError(' min value of rolls_win ==1  ')
	check_qulify(df,dateCol,rollBy)

# 	df = df.sort_values(dateCol).reset_index(drop=True)
	for day in rolls_win:
		tmp_shifts = []
		for rol in range(1,day+1):
			thshif = shif_win + rol-1
			tm = 'shif_{}_num'.format( thshif )
			df[tm] = df.groupby([rollBy])[rollOn].shift( thshif )
			tmp_shifts.append( tm )
		if day == 2:
			feat = 'base_{}_rolling_{}_mean_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].mean(axis=1)
		else:
			feat = 'base_{}_rolling_{}_mean_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].mean(axis=1)
			feat = 'base_{}_rolling_{}_max_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].max(axis=1)
			feat = 'base_{}_rolling_{}_min_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].min(axis=1)
			feat = 'base_{}_rolling_{}_median_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].median(axis=1)
			feat = 'base_{}_rolling_{}_skew_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].skew(axis=1)
		df.drop(tmp_shifts,axis=1,inplace=True)
	return df


# In[17]:



data = pd.read_csv('kpi_train.csv')

test = pd.read_csv('kpi_test.csv')
test['pv'] = np.nan
test['uv'] = np.nan
data = data.append(test).reset_index(drop=True)


# In[18]:


def past_m_summary(df,shif_win,rollBy,rollOn,dayRnge,dateCol):
    tmp_shifts = []
    use_cols = [rollBy,rollOn,dateCol]
    result = pd.DataFrame()
    minDy = dayRnge[0]
    maxDy = dayRnge[1]
    for day in range(minDy,maxDy+1):
        _df = df[df[dateCol]==day][use_cols]
        if day-shif_win< 2:
            pass
        elif day-shif_win== 2:
            dfs = df[df[dateCol] <= day-shif_win]
            gp = dfs.groupby([rollBy])[rollOn].mean()
            _df['Past_{}_Summ_{}_{}'.format( shif_win,rollOn,'mean' )] = _df[rollBy].map(gp)
        else:
            dfs = df[df[dateCol] <= day-shif_win]
            gp = dfs.groupby([rollBy])[rollOn].mean()
            _df['Past_{}_Summ_{}_{}'.format( shif_win,rollOn,'mean' )] = _df[rollBy].map(gp)
            gp = dfs.groupby([rollBy])[rollOn].median()
            _df['Past_{}_Summ_{}_{}'.format( shif_win,rollOn,'median' )] = _df[rollBy].map(gp)
            gp = dfs.groupby([rollBy])[rollOn].skew()
            _df['Past_{}_Summ_{}_{}'.format( shif_win,rollOn,'skew' )] = _df[rollBy].map(gp)
        result = pd.concat([result,_df])
            
    del result[rollOn]
    del _df
    df = df.merge(result,on=[rollBy,dateCol],how='left' )
    del result
    return df
    


# In[19]:


target = 'pv'
type_ = '$pageview'
# reg_input_success
# '$pageview'
# 20190304
usedata = data[data.date>=20190304].reset_index(drop=True)
reg_input_data = usedata[usedata.event_type == type_ ][['date',target]]
reg_input_data['dow'] = pd.to_datetime(reg_input_data['date'].map(str)).dt.dayofweek
reg_input_data['weekOfYear'] = pd.to_datetime(reg_input_data['date'].map(str)).dt.weekofyear
reg_input_data = reg_input_data[reg_input_data.weekOfYear>=10]
reg_input_data['weekOfYear'] = reg_input_data['weekOfYear']-9


df = reg_input_data.copy()
df = df.sort_values('date').reset_index(drop=True)
df['dow_dum_'] = df['dow'].copy()
df = pd.get_dummies(df,columns=['dow_dum_'])

sorted_unique_date = sorted(df.date.unique().tolist())
date_order_dict = dict( zip( sorted_unique_date, range(1,len(sorted_unique_date)+1 ) ) )
df['dayIndex'] = df['date'].map(date_order_dict)
df = df.sort_values('dayIndex').reset_index(drop = True)
df['none'] = 1
dayRnge = [df.dayIndex.min(), df.dayIndex.max()]
for shi in [7,10,14]:
    df = past_m_summary(df,shi,'none',target,dayRnge,'dayIndex')


# df['is_monday'] = 0
# df.loc[df['dow']==0,'is_monday'] = 1

rollBy = 'dow'
rollOn = target
shif_wins = [1,2,3]
rolls_win = [2,3]
for shif_win in shif_wins:
    df = My_rolling( df , rollBy,rollOn,shif_win,rolls_win,'weekOfYear' )
df['weekday_df'] = df.groupby(['dow'])[target].diff()
df['weekday_df'] = df.groupby(['dow'])['weekday_df'].shift(1)
base_day = df.weekOfYear.max()
df = df.fillna( df.mean())
# df.loc[df.dow == 0,'dow']=np.nan


# train = train[train.date<20190503].reset_index(drop=True)
# train


# In[20]:


# df.loc[df.dow == 0,'dow']=np.nan
test = df.loc[ df.weekOfYear==base_day ].reset_index(drop=True)
valid = df.loc[ df.weekOfYear.isin([base_day-2,base_day-1]) ].reset_index(drop=True)

train = df.loc[ df.weekOfYear<base_day-2 ].reset_index(drop=True)
print(train.date.max())


# In[21]:


feats = [c for c in df.columns  if c not in ['date','weekOfYear','pv']]


# In[22]:



params_in = {'num_leaves': 16,
        #   'min_child_weight': 0.044,
          'feature_fraction': 0.8,
          'bagging_fraction': 0.8,
                 'bagging_freq': 10,
          'min_data_in_leaf': 14,
          'objective': 'regression',
          'max_depth': -1,
          'learning_rate': 0.07,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          'metric': 'rmse',
          'num_threads': 10,
          "verbosity": -1,
          'reg_alpha': 0.038,
          'reg_lambda': 0.5,
          'random_state': 88
         }

feats = [
    'dow',
    # 'dayIndex',
'weekOfYear',
'weekday_df',
'dow_dum__0',
#  'dow_dum__1',
#  'dow_dum__2',
#  'dow_dum__3',
#  'dow_dum__4',
#  'dow_dum__5',
#  'dow_dum__6',
# 'base_2_rolling_3_median_pv_OF_dow',
'base_1_rolling_2_mean_pv_OF_dow',
# 'base_1_rolling_3_mean_pv_OF_dow', 
'base_1_rolling_3_max_pv_OF_dow',
'base_1_rolling_3_median_pv_OF_dow',
# 'base_2_rolling_2_mean_pv_OF_dow',
# 'Past_7_Summ_pv_skew',
'Past_7_Summ_pv_mean',
# 'Past_7_Summ_pv_median',
# 'Past_10_Summ_pv_mean',
'base_1_rolling_3_min_pv_OF_dow',
'base_2_rolling_3_min_pv_OF_dow',
# 'base_2_rolling_3_mean_pv_OF_dow',
'base_2_rolling_3_skew_pv_OF_dow',
'base_3_rolling_3_min_pv_OF_dow',
'base_3_rolling_3_mean_pv_OF_dow',
 'base_3_rolling_3_max_pv_OF_dow',
 'base_3_rolling_3_median_pv_OF_dow',
#  'base_3_rolling_3_skew_pv_OF_dow',
 
'Past_10_Summ_pv_mean',
#  'Past_10_Summ_pv_median',
#  'Past_10_Summ_pv_skew',
#  'Past_14_Summ_pv_mean',
#  'Past_14_Summ_pv_median',
#  'Past_14_Summ_pv_skew',


# 'base_3_rolling_3_skew_pv_OF_dow',
'base_3_rolling_3_median_pv_OF_dow',
'base_3_rolling_3_mean_pv_OF_dow',
'base_3_rolling_2_mean_pv_OF_dow',
]

pv_pred,gbm =  model_logged_npmse_predict(params_in,train,valid,test,feats)
valid_pred  = gbm.predict(valid[feats])
weight = correction(valid_pred,np.log1p(valid[target]))
#  [15]	valid_0's l1: 8.75314	valid_0's npmse: 0.271388
pv_pre =np.expm1( pv_pred*weight)


# In[23]:


import pandas as pd
import numpy as np
#Import packages for preprocessing (data cleaning)
from sklearn.preprocessing import RobustScaler, LabelEncoder
#Import packages for data visualisation
import matplotlib.pyplot as plt
#Import packages for model testing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
#Import packages for modeling
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline
data_to_train = train.append(valid).reset_index(drop=True)
labels_to_use = data_to_train[target]
data_to_train = data_to_train[feats]

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0007, random_state=1))
ridge = make_pipeline(RobustScaler(), Ridge(alpha =0.0007, random_state=42))
def evaluation(model):
    result= np.sqrt(-cross_val_score(model, data_to_train, labels_to_use, cv = 5, scoring = 'neg_mean_squared_error'))
    return(result)
ls_score = evaluation(lasso)
print("Lasso score: {:.5f}\n".format(ls_score.mean()))


# In[24]:


ridge.fit(train[feats ], np.log1p(train[target]))
test_pred_1 = ridge.predict(test[ feats])
valid_pred  = ridge.predict(valid[feats])
weight = correction(valid_pred,np.log1p(valid[target]))
pre2 = np.expm1(test_pred_1*weight)


# In[25]:


feats =['dow',
 'base_1_rolling_3_mean_pv_OF_dow',
 'weekday_df',
 'base_1_rolling_3_max_pv_OF_dow',
 'dow_dum__4',
 'dow_dum__3',
 'dow_dum__6',
 'dayIndex',
 'Past_10_Summ_pv_skew',
 'Past_14_Summ_pv_skew',
 'none']
ridge.fit(train[feats ], np.log1p(train[target]))
test_pred_1 = ridge.predict(test[ feats])
valid_pred  = ridge.predict(valid[feats])
weight = correction(valid_pred,np.log1p(valid[target]))
pre2 = np.expm1(test_pred_1*weight)


# In[26]:


besti=80
bestj=20
r_n = (besti*pre2 + (np.array(pv_pre)*(bestj)))/100
pageview_pv =r_n
    


# In[27]:


pageview_pv


# In[28]:


import pandas as pd
import numpy as np
import os
import warnings
import time
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import lightgbm as lgb
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm, tqdm_notebook
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import math
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
# from gensim.models import Word2Vec
import json
import gc
import re
warnings.filterwarnings('ignore')


import lightgbm as lgb

def eval_score(preds, train_data):
    labels = train_data.get_label()
    df = pd.DataFrame()
    df['y_true'] = labels
    df['y_pred'] = preds
    df['sc'] = ((df['y_pred'] - df['y_true'] )/df['y_true']) ** 2
    score = np.sqrt(df.sc.mean())
    
    return 'npmse', score, False

def score(preds, labels):
    df = pd.DataFrame()
    df['y_true'] = labels
    df['y_pred'] = preds
    df['sc'] = ((df['y_pred'] - df['y_true'] )/df['y_true']) ** 2
    sc = np.sqrt(df.sc.mean())
    
    return sc


def model_npmse_predict(params_in,train,valid,test,pred1):
    'predict_in'
#     params_in ={
#         'learning_rate': 0.01,
#         'boosting_type': 'gbdt',
#         'objective': 'regression',
#         'metric': 'mae',
#         'num_leaves': 400,
#         'feature_fraction': 0.7,
#         'bagging_fraction': 0.9,
#         'bagging_freq': 10,
#         'seed': 666,
#         'bagging_seed': 10001,
#         'feature_fraction_seed': 7,
#         'min_data_in_leaf': 20,
#         'reg_sqrt': True,
#         'nthread': 8,
#         'verbose': -1,
#     }
    
    
    
# ,categorical_feature=cateCols
# ,categorical_feature=cateCols
    dtrain = lgb.Dataset(train[feats],free_raw_data =False,  label=train[target])
    dvalid = lgb.Dataset(valid[feats], free_raw_data =False, label=valid[target])
    gbm = lgb.train(
  	 params=params_in,
  	 train_set=dtrain,
  	 num_boost_round=500,
  	 valid_sets=[dvalid],
         early_stopping_rounds=30,
    	 feval = eval_score,
    	 verbose_eval=10
	)
    
    pred = gbm.predict(test[pred1])
    
    return pred,gbm


    


def model_logged_npmse_predict(params_in,train,valid,test,pred1):
    'predict_in'
#     params_in ={
#         'learning_rate': 0.01,
#         'boosting_type': 'gbdt',
#         'objective': 'regression',
#         'metric': 'mae',
#         'num_leaves': 400,
#         'feature_fraction': 0.7,
#         'bagging_fraction': 0.9,
#         'bagging_freq': 10,
#         'seed': 666,
#         'bagging_seed': 10001,
#         'feature_fraction_seed': 7,
#         'min_data_in_leaf': 20,
#         'reg_sqrt': True,
#         'nthread': 8,
#         'verbose': -1,
#     }
    
    
    
# ,categorical_feature=cateCols
# ,categorical_feature=cateCols
    dtrain = lgb.Dataset(train[feats],free_raw_data =False,  label=np.log1p(train[target]))
    dvalid = lgb.Dataset(valid[feats], free_raw_data =False, label=np.log1p(valid[target]))
    gbm = lgb.train(
  	 params=params_in,
  	 train_set=dtrain,
  	 num_boost_round=1000,
  	 valid_sets=[dvalid],
         early_stopping_rounds=30,
    	 feval = eval_score,
    	 verbose_eval=10
	)
    
    pred = gbm.predict(test[pred1])
    
    return pred,gbm



from sklearn.metrics import mean_absolute_error as mae

def correction(preds,labels):
    weights = np.arange(0.5, 1.02, 0.005)
    errors = []
    for w in weights:
        
        sc = score(preds*w, labels)
        errors.append(sc)
    idx = errors.index(min(errors))
    print('Best weight is {}, mae Score is {:.4f}'.format(weights[idx], min(errors)))
    return weights[idx]
import os

import matplotlib.pyplot as plt
import seaborn as sns
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by = "importance", ascending = False)[:50
                                                                                                                                        ].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    # print(feature_importance_df_[feature_importance_df_.feature.map(lambda x:'IS' in x)])
    # print(best_features.feature.tolist())
    plt.figure(figsize = (15, 10))
    sns.barplot(x = "importance", y = "feature", data = best_features.sort_values(by = "importance", ascending = False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
#     plt.savefig('../cache/lgbm_importances.png')
def get_Impdf(lgb_model,feats,fileIndex ):
    feature_importance_df = pd.DataFrame()
    feature_importance_df['feature'] = feats
    feature_importance_df['importance'] = lgb_model.feature_importance()
    # imp_df[imp_df.fea.map(lambda x:'ip_first' not in x)==True].sort_values(by=['imp'], ascending=False).tail(90)
    import datetime
    td = datetime.datetime.today()
    tdstr = str(td)[:10]
    # save_imp_csv = '../cache/FeaImp_{}_{}.csv'.format( tdstr,fileIndex )
    display_importances(feature_importance_df)


def check_qulify(df,dateCol,rollBy):
	gp = df.groupby([ dateCol,rollBy ] )[ dateCol ].agg({'count':'count'}).reset_index()
	if gp['count'].nunique()!=1:
		raise TypeError(' rollBy ERROR!!! combination of dateCol and rollBy must be unique')
	del gp



def My_rolling( df , rollBy,rollOn,shif_win,rolls_win,dateCol ):
	if isinstance(rolls_win,list):
		pass
	else:
		raise TypeError(' rolls_win must be a list  ')
	if min(rolls_win)==1:
		raise TypeError(' min value of rolls_win ==1  ')
	check_qulify(df,dateCol,rollBy)

# 	df = df.sort_values(dateCol).reset_index(drop=True)
	for day in rolls_win:
		tmp_shifts = []
		for rol in range(1,day+1):
			thshif = shif_win + rol-1
			tm = 'shif_{}_num'.format( thshif )
			df[tm] = df.groupby([rollBy])[rollOn].shift( thshif )
			tmp_shifts.append( tm )
		if day == 2:
			feat = 'base_{}_rolling_{}_mean_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].mean(axis=1)
		else:
			feat = 'base_{}_rolling_{}_mean_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].mean(axis=1)
			feat = 'base_{}_rolling_{}_max_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].max(axis=1)
			feat = 'base_{}_rolling_{}_min_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].min(axis=1)
			feat = 'base_{}_rolling_{}_median_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].median(axis=1)
			feat = 'base_{}_rolling_{}_skew_{}_OF_{}'.format(  shif_win,day,rollOn,rollBy )
			df[feat] = df[tmp_shifts].skew(axis=1)
		df.drop(tmp_shifts,axis=1,inplace=True)
	return df


# In[29]:



data = pd.read_csv('kpi_train.csv')

test = pd.read_csv('kpi_test.csv')
test['pv'] = np.nan
test['uv'] = np.nan
data = data.append(test).reset_index(drop=True)


# In[30]:


def past_m_summary(df,shif_win,rollBy,rollOn,dayRnge,dateCol):
    tmp_shifts = []
    use_cols = [rollBy,rollOn,dateCol]
    result = pd.DataFrame()
    minDy = dayRnge[0]
    maxDy = dayRnge[1]
    for day in range(minDy,maxDy+1):
        _df = df[df[dateCol]==day][use_cols]
        if day-shif_win< 2:
            pass
        elif day-shif_win== 2:
            dfs = df[df[dateCol] <= day-shif_win]
            gp = dfs.groupby([rollBy])[rollOn].mean()
            _df['Past_{}_Summ_{}_{}'.format( shif_win,rollOn,'mean' )] = _df[rollBy].map(gp)
        else:
            dfs = df[df[dateCol] <= day-shif_win]
            gp = dfs.groupby([rollBy])[rollOn].mean()
            _df['Past_{}_Summ_{}_{}'.format( shif_win,rollOn,'mean' )] = _df[rollBy].map(gp)
            gp = dfs.groupby([rollBy])[rollOn].median()
            _df['Past_{}_Summ_{}_{}'.format( shif_win,rollOn,'median' )] = _df[rollBy].map(gp)
            gp = dfs.groupby([rollBy])[rollOn].skew()
            _df['Past_{}_Summ_{}_{}'.format( shif_win,rollOn,'skew' )] = _df[rollBy].map(gp)
        result = pd.concat([result,_df])
            
    del result[rollOn]
    del _df
    df = df.merge(result,on=[rollBy,dateCol],how='left' )
    del result
    return df
    


# In[31]:


target = 'uv'
type_ = '$pageview'
# reg_input_success
# '$pageview'
# 20190304
usedata = data[data.date>=20190304].reset_index(drop=True)
reg_input_data = usedata[usedata.event_type == type_ ][['date',target]]
reg_input_data['dow'] = pd.to_datetime(reg_input_data['date'].map(str)).dt.dayofweek
reg_input_data['weekOfYear'] = pd.to_datetime(reg_input_data['date'].map(str)).dt.weekofyear
reg_input_data = reg_input_data[reg_input_data.weekOfYear>=10]
reg_input_data['weekOfYear'] = reg_input_data['weekOfYear']-9


df = reg_input_data.copy()
df = df.sort_values('date').reset_index(drop=True)
df['dow_dum_'] = df['dow'].copy()
df = pd.get_dummies(df,columns=['dow_dum_'])

sorted_unique_date = sorted(df.date.unique().tolist())
date_order_dict = dict( zip( sorted_unique_date, range(1,len(sorted_unique_date)+1 ) ) )
df['dayIndex'] = df['date'].map(date_order_dict)
df = df.sort_values('dayIndex').reset_index(drop = True)
df['none'] = 1
dayRnge = [df.dayIndex.min(), df.dayIndex.max()]
for shi in [7,10,14]:
    df = past_m_summary(df,shi,'none',target,dayRnge,'dayIndex')


# df['is_monday'] = 0
# df.loc[df['dow']==0,'is_monday'] = 1

rollBy = 'dow'
rollOn = target
shif_wins = [1,2,3]
rolls_win = [2,3]
for shif_win in shif_wins:
    df = My_rolling( df , rollBy,rollOn,shif_win,rolls_win,'weekOfYear' )
df['weekday_df'] = df.groupby(['dow'])[target].diff()
df['weekday_df'] = df.groupby(['dow'])['weekday_df'].shift(1)
base_day = df.weekOfYear.max()
df = df.fillna( df.mean())
# df.loc[df.dow == 0,'dow']=np.nan


# train = train[train.date<20190503].reset_index(drop=True)
# train


# In[32]:


# df.loc[df.dow == 0,'dow']=np.nan
test = df.loc[ df.weekOfYear==base_day ].reset_index(drop=True)
valid = df.loc[ df.weekOfYear.isin([base_day-2,base_day-1]) ].reset_index(drop=True)

train = df.loc[ df.weekOfYear<base_day-2 ].reset_index(drop=True)
print(train.date.max())


# In[33]:


feats = [c for c in df.columns  if c not in ['date','weekOfYear','pv']]


# In[34]:



params_in = {'num_leaves': 16,
        #   'min_child_weight': 0.044,
          'feature_fraction': 0.8,
          'bagging_fraction': 0.8,
                 'bagging_freq': 10,
          'min_data_in_leaf': 14,
          'objective': 'regression',
          'max_depth': -1,
          'learning_rate': 0.07,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          'metric': 'rmse',
          'num_threads': 10,
          "verbosity": -1,
          'reg_alpha': 0.038,
          'reg_lambda': 0.5,
          'random_state': 88
         }




# In[35]:


import pandas as pd
import numpy as np
#Import packages for preprocessing (data cleaning)
from sklearn.preprocessing import RobustScaler, LabelEncoder
#Import packages for data visualisation
import matplotlib.pyplot as plt
#Import packages for model testing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
#Import packages for modeling
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline
data_to_train = train.append(valid).reset_index(drop=True)
labels_to_use = data_to_train[target]
data_to_train = data_to_train[feats]

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0007, random_state=1))
ridge = make_pipeline(RobustScaler(), Ridge(alpha =0.0007, random_state=42))
def evaluation(model):
    result= np.sqrt(-cross_val_score(model, data_to_train, labels_to_use, cv = 5, scoring = 'neg_mean_squared_error'))
    return(result)
ls_score = evaluation(lasso)
print("Lasso score: {:.5f}\n".format(ls_score.mean()))


# In[36]:


feats=['dow',
 'weekOfYear',
 'weekday_df',
 'base_1_rolling_2_mean_uv_OF_dow',
 'dow_dum__3',
 'Past_10_Summ_uv_skew',
 'dow_dum__4',
 'dow_dum__6',
 'dow_dum__1',
 'base_1_rolling_3_min_uv_OF_dow',
 'dayIndex',
 'none']


# In[37]:



ridge.fit(train[feats ], np.log1p(train[target]))
test_pred_1 = ridge.predict(test[ feats])
valid_pred  = ridge.predict(valid[feats])
weight = correction(valid_pred,np.log1p(valid[target]))
pre2 = np.expm1(test_pred_1*weight)


# In[38]:


pageview_uv = pre2


# In[39]:


print('$pageview:PV',pageview_pv)
print('$pageview:UV',pageview_uv)
print('reg_input_success:PV',reg_pv)
print('reg_input_success:UV',reg_uv)


# In[40]:


#sub = pd.read_csv('kpi_test.csv')


# In[41]:


#global index_page
#index_page= 0


# In[42]:


'''
def fun(x,y,e_type):
    global index_page
    if x==e_type:
        result = pageview_uv[index_page]
        index_page = index_page+1
        return result
    else:
        return y
'''


# In[43]:


#sub['pv'] = sub.apply(lambda rows:fun(rows['event_type'],rows['pv'],'$pageview'),axis=1)


# In[44]:


#sub['uv'] = sub.apply(lambda rows:fun(rows['event_type'],rows['uv'],'$pageview'),axis=1)


# In[45]:


#sub['pv'] = sub.apply(lambda rows:fun(rows['event_type'],rows['pv'],'reg_input_success'),axis=1)


# In[46]:


#sub['uv'] = sub.apply(lambda rows:fun(rows['event_type'],rows['uv'],'reg_input_success'),axis=1)


# In[47]:


#col = ['date','pv','uv']
#sub[col] = sub[col].astype(int)


# In[48]:


#sub


# In[49]:


#sub.to_csv('result_haha.csv',index = False)


# In[50]:


print('$pageview:PV',pageview_pv)
print('$pageview:UV',pageview_uv)
print('reg_input_success:PV',reg_pv)
print('reg_input_success:UV',reg_uv)

