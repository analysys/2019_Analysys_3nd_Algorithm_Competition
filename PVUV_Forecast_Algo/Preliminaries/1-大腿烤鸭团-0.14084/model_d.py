import pandas as pd
import numpy as np
from config import cfg, logger
from processing.dataset_generator import gen_train_test
from models.model import MultiOutputTsModel
from fileio.df_tools import SubmissionConvertor
from sklearn.linear_model import BayesianRidge


def model_d(train, old_df):
    final_df = old_df.copy()
    logger.info('===STAGE D===')
    logger.info('Splitting dataset...')
    train_x, train_y, test_x = gen_train_test(train)

    sc = SubmissionConvertor()

    test_y = cfg.submit.sample
    test_y = sc.sub2df(test_y)

    """Base Model D - Bayesian Ridge - Pageview PV"""
    logger.info('Start training base model D...')
    power_br = BayesianRidge(
        normalize=True, alpha_1=1e5, lambda_1=0, lambda_2=0
    )
    power_br_features = [
        'dayofweek', 'weekend', 'userflow', 'pg_pv_lastweek', 'reg_pv_lastweek',
        'pg_uv_lastweek', 'reg_uv_lastweek', 'pg_pv_diff_lastweek',
        'reg_pv_diff_lastweek', 'pg_uv_diff_lastweek', 'reg_uv_diff_lastweek',
        'pg_pv_last2week', 'reg_pv_last2week', 'pg_uv_last2week',
        'reg_uv_last2week', 'pg_pv_diff_last2week', 'reg_pv_diff_last2week',
        'pg_uv_diff_last2week',  'pg_pv_sum_last2week',
        'reg_pv_sum_last2week', 'pg_uv_sum_last2week', 'reg_uv_sum_last2week'
    ]
    mts_power_br = MultiOutputTsModel(
        power_br, train_x[power_br_features], train_y,
        test_x[power_br_features], test_y
    )
    mts_power_br.fit()
    pred_y_power_br = mts_power_br.predict()
    logger.info('Inference completed.')

    final_df.loc[:, ['pg_pv']] = pred_y_power_br[['pg_pv']].values

    logger.info('End!')
    return final_df, pred_y_power_br
