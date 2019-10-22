import pandas as pd
import numpy as np
from config import cfg, logger
from processing.dataset_generator import gen_train_test
from models.statespace import StateSpaceModel
from models.model import MultiOutputTsModel
from fileio.df_tools import SubmissionConvertor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import BayesianRidge


def model_c(train, old_df):
    final_df = old_df.copy()
    logger.info('===STAGE C===')
    logger.info('Splitting dataset...')
    train_x, train_y, test_x = gen_train_test(train)

    sc = SubmissionConvertor()

    test_y = cfg.submit.sample
    test_y = sc.sub2df(test_y)

    """Base Model C1 - SARIMAX"""
    logger.info('Start training base model C1...')
    sarima = StateSpaceModel(
        SARIMAX, order=(3, 2, 14), seasonal_order=(2, 1, 1, 7)
    )
    mts_sarima = MultiOutputTsModel(
        sarima, train_x[['userflow']], train_y,
        test_x[['userflow']], test_y
    )
    mts_sarima.fit(disp=0)
    pred_y_sarima = mts_sarima.predict(
        start=0, end=len(train_x) + 6,
        exog=test_x[['userflow']]
    )
    logger.info('Inference completed.')


    resid_train_y = (train_y - pred_y_sarima).dropna().iloc[3:, :].reset_index(drop=True)
    resid_train_x = train_x.iloc[3:, :].reset_index(drop=True)

    """Base Model C2 - Bayesian Ridge with parital info."""
    logger.info('Start training base model C2...')
    partial_br = BayesianRidge(
        normalize=True,
        n_iter=200,
        alpha_1=0,
        alpha_2=0,
        lambda_1=0,
        lambda_2=1e6
    )
    br_features = [
        'dayofweek', 'weekend', 'userflow', 'pg_pv_lastweek', 'reg_pv_lastweek',
        'pg_uv_lastweek', 'reg_uv_lastweek', 'pg_pv_diff_lastweek',
        'reg_pv_diff_lastweek', 'pg_uv_diff_lastweek', 'reg_uv_diff_lastweek',
        'pg_pv_last2week', 'reg_pv_last2week', 'pg_uv_last2week',
        'reg_uv_last2week', 'pg_pv_diff_last2week', 'reg_pv_diff_last2week',
        'pg_uv_diff_last2week', 'reg_uv_diff_last2week', 'pg_pv_sum_last2week',
        'reg_pv_sum_last2week', 'pg_uv_sum_last2week', 'reg_uv_sum_last2week',
        'pg_pv_lag_0_mean', 'reg_pv_lag_0_mean', 'pg_uv_lag_0_mean', 'reg_uv_lag_0_mean'
    ]
    partial_info = 119
    mts_partial_br = MultiOutputTsModel(
        partial_br,
        resid_train_x[br_features].iloc[-partial_info:, :],
        resid_train_y.iloc[-partial_info:, :],
        test_x[br_features], test_y
    )
    mts_partial_br.fit()
    pred_y_partial_br = mts_partial_br.predict()
    pred_y_partial_br += pred_y_sarima.iloc[-7:].values
    logger.info('Inference completed.')


    sc = SubmissionConvertor()

    final_df.loc[:, ['pg_uv']] = pred_y_partial_br[['pg_uv']].values
    final_sub = sc.df2sub(final_df)

    logger.info('End!')
    return final_df, pred_y_partial_br
