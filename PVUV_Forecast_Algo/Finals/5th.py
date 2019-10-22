
# coding: utf-8


get_ipython().magic('cd easyobserve/')


# # 工具库导入


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
from sklearn.metrics import mean_absolute_error as mae
import json
import gc
import re
import os

#Import packages for preprocessing (data cleaning)
from sklearn.preprocessing import RobustScaler, LabelEncoder
#Import packages for modeling
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore')


# # 1.评估函数(之后的模型需要用上)
# # 2.用于特征提取的元工具函数


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

def correction(preds,labels):
    weights = np.arange(0.5, 1.02, 0.005)
    errors = []
    for w in weights:
        
        sc = score(preds*w, labels)
        errors.append(sc)
    idx = errors.index(min(errors))
    print('Best weight is {}, mae Score is {:.4f}'.format(weights[idx], min(errors)))
    return weights[idx]

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
    


# # 1.特征提取函数 
# # 2.训练集与测试集拆分函数 
# # 3.模型训练与预测函数



def extract_feature(data,type_,target):
    usedata = data.reset_index(drop=True).copy()
    temp = usedata[usedata.event_type == type_ ][['date',target]]
    temp['dow'] = pd.to_datetime(temp['date'].map(str)).dt.dayofweek
    temp['weekOfYear'] = pd.to_datetime(temp['date'].map(str)).dt.weekofyear
    temp = temp[temp.weekOfYear>=10]
    temp['weekOfYear'] = temp['weekOfYear']-9


    df = temp.copy()
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

    rollBy = 'dow'
    rollOn = target
    shif_wins = [1,2,3]
    rolls_win = [2,3]
    for shif_win in shif_wins:
        df = My_rolling( df , rollBy,rollOn,shif_win,rolls_win,'weekOfYear' )
    df['weekday_df'] = df.groupby(['dow'])[target].diff()
    df['weekday_df'] = df.groupby(['dow'])['weekday_df'].shift(1)
    week = df.weekOfYear.max()
    df = df.fillna( df.mean())
    return df,week
def train_test_split(data,base_day):
    test = data.loc[ data.weekOfYear==base_day ].reset_index(drop=True)
    valid = data.loc[ data.weekOfYear.isin([base_day-2,base_day-1]) ].reset_index(drop=True)
    train = data.loc[ data.weekOfYear<base_day-2 ].reset_index(drop=True)
    return train,valid,test
def model_train_predict(model,train,valid,test,feats,target):
    print('train shape',train.shape,'test shape',test.shape)
    model.fit(train[feats ], np.log1p(train[target]))
    test_pred_1 = model.predict(test[feats])
    valid_pred  = model.predict(valid[feats])
    weight = correction(valid_pred,np.log1p(valid[target]))
    pre2 = np.expm1(test_pred_1*weight)
    return pre2


# # 导入数据 训练集和测试集



original_data = pd.read_csv('kpi_train_new.csv')
sub = pd.read_csv('kong.csv')


# # $pageview pv指标



#数据准备
target = 'pv'
type_ = '$pageview'
data=original_data[-168-14:]
test = sub.copy()
test['pv'] = np.nan
test['uv'] = np.nan
data = data.append(test).reset_index(drop=True)

#特征提取
df,week_day = extract_feature(data,type_,target)
#训练集验证集测试集拆分
train,valid,test = train_test_split(df,week_day)

#模型参数
ridge = make_pipeline(RobustScaler(), Ridge(alpha =0.0007, random_state=42))
#特征选择
feats =['dow',
 'base_1_rolling_3_mean_pv_OF_dow',
 'Past_14_Summ_pv_skew',
 'base_3_rolling_3_skew_pv_OF_dow',
 'base_2_rolling_3_skew_pv_OF_dow',
 'dow_dum__0',
 'dow_dum__1',
 'Past_7_Summ_pv_mean',
 'none']
#模型训练与预测
pageview_pv = model_train_predict(ridge,train,valid,test,feats,target)
pageview_pv


# # $pageview uv指标


#数据准备
target = 'uv'
type_ = '$pageview'
data=original_data[-168-14:]
test = sub.copy()
test['pv'] = np.nan
test['uv'] = np.nan
data = data.append(test).reset_index(drop=True)
#特征提取
df,week_day = extract_feature(data,type_,target)
#训练集验证集测试集拆分
train,valid,test = train_test_split(df,week_day)

#模型参数
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0007, random_state=1))
#特征选择
feats=['dow',
 'base_1_rolling_3_median_uv_OF_dow',
 'dow_dum__5',
 'Past_10_Summ_uv_mean',
 'dow_dum__6',
 'Past_7_Summ_uv_mean',
 'weekday_df',
 'dow_dum__1',
 'base_3_rolling_3_skew_uv_OF_dow',
 'none']
#模型训练与预测
pageview_uv = model_train_predict(lasso,train,valid,test,feats,target)
pageview_uv


# # reg_input_success  pv指标



#数据准备
target = 'pv'
type_ = 'reg_input_success'

data=original_data[-168-96+14:]
test = sub.copy()
data = data.append(test).reset_index(drop=True)
#特征提取
df,week_day = extract_feature(data,type_,target)
#训练集验证集测试集拆分
train,valid,test = train_test_split(df,week_day)

#模型参数
ridge = make_pipeline(RobustScaler(), Ridge(alpha =0.0007, random_state=42))
#特征选择
feats=['dow',
 'base_1_rolling_3_mean_pv_OF_dow',
 'dow_dum__5',
 'dow_dum__6',
 'base_2_rolling_3_median_pv_OF_dow',
 'base_3_rolling_3_min_pv_OF_dow',
 'base_3_rolling_3_skew_pv_OF_dow',
 'Past_14_Summ_pv_skew',
 'dow_dum__4',
 'base_1_rolling_2_mean_pv_OF_dow',
 'dow_dum__2',
 'weekday_df',
 'dow_dum__3',
 'none']
#模型训练与预测
reg_pv = model_train_predict(ridge,train,valid,test,feats,target)
reg_pv


# # reg_input_success  uv指标



#数据准备
target = 'uv'
type_ = 'reg_input_success'

data=original_data[-168-96+14:]
test = sub.copy()
data = data.append(test).reset_index(drop=True)

#特征提取
df,week_day = extract_feature(data,type_,target)
#训练集验证集测试集拆分
train,valid,test = train_test_split(df,week_day)

#模型参数
ridge = make_pipeline(RobustScaler(), Ridge(alpha =0.0007, random_state=42))
#特征选择
feats=['dow',
 'base_1_rolling_2_mean_uv_OF_dow',
 'dow_dum__5',
 'base_2_rolling_3_median_uv_OF_dow',
 'base_3_rolling_3_median_uv_OF_dow',
 'dow_dum__3',
 'dow_dum__6',
 'base_3_rolling_3_min_uv_OF_dow',
 'base_3_rolling_3_skew_uv_OF_dow',
 'dow_dum__4',
 'Past_10_Summ_uv_median',
 'base_2_rolling_2_mean_uv_OF_dow',
 'base_3_rolling_2_mean_uv_OF_dow',
 'Past_14_Summ_uv_skew',
 'weekday_df',
 'none']
#模型训练与预测
reg_uv = model_train_predict(ridge,train,valid,test,feats,target)
reg_uv


# # 四个指标打印输出


pageview_pv = [int(v) for v in pageview_pv]
pageview_uv = [int(v) for v in pageview_uv]
reg_pv = [int(v) for v in reg_pv]
reg_uv = [int(v) for v in reg_uv]




print('$pageview:PV',pageview_pv)
print('$pageview:UV',pageview_uv)
print('reg_input_success:PV',reg_pv)
print('reg_input_success:UV',reg_uv)

