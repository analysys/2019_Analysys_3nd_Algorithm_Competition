import pandas as pd
import numpy as np
from config import cfg, logger
from fileio.data_reader import read_files
from processing.preprocessing import nan_clean, DataPreparation, NoiseCancellation
from processing.feature import FeatureExtraction
from processing.dataset_generator import gen_train_test_weekend
from processing.dataset_generator import gen_train_test_weekend_cross
from models.statespace import StateSpaceModel
from models.simple_filling import SimpleFilling
from models.model import TsModel, MultiOutputTsModel
from utils.submission import submit_export
from fileio.df_tools import SubmissionConvertor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import ElasticNetCV
import lightgbm as lgb
from models.fusion import WeightedAveraging


def model_b(train, model_a_df):
    final_df = model_a_df.copy()
    logger.info('===STAGE B===')
    logger.info('Splitting dataset...')
    train_x, train_y, test_x = gen_train_test_weekend(train)

    sc = SubmissionConvertor()

    test_y = cfg.submit.sample
    test_y = sc.sub2df(test_y).iloc[:5]

    """Base Model B1 - SARIMAX"""
    logger.info('Start training base model B1...')
    sarima = StateSpaceModel(
        SARIMAX, order=(3, 2, 10), seasonal_order=(2, 1, 1, 5)
    )
    mts_sarima = MultiOutputTsModel(
        sarima, train_x[['userflow']], train_y,
        test_x[['userflow']], test_y
    )
    mts_sarima.fit(disp=0)
    pred_y_sarima = mts_sarima.predict(
        start=0, end=len(train_x) + 4,
        exog=test_x[['userflow']]
    )
    logger.info('Inference completed.')

    # Final dataframe
    # final_df.iloc[:5, 1] = pred_y_sarima.iloc[-5:, 1].astype(int).values
    # final_df.iloc[:5, 3] = pred_y_sarima.iloc[-5:, 3].astype(int).values


    # """Base Model B2 - LGB"""
    # logger.info('Start training base model B2...')
    # lgb_model = lgb.LGBMRegressor(
    #     boosting_type='gbdt',
    #     num_leaves=24,
    #     reg_alpha=1,
    #     reg_lambda=2,
    #     max_depth=6,
    #     learning_rate=0.01,
    #     min_child_samples=3,
    #     n_estimators=1000,
    #     subsample=0.6,
    #     colsample_bytree=0.8,
    #     random_state=233,
    #     silent=True
    # )
    # mts_lgb = MultiOutputTsModel(
    #     lgb_model, train_x, train_y, test_x, test_y
    # )
    # mts_lgb.fit()
    # pred_y_lgb = mts_lgb.predict()
    # logger.info('Inference completed.')


    """Base Model B3 - ElasticNetCV"""
    logger.info('Start training base model B2...')
    elnetcv_model = ElasticNetCV(cv=3, random_state=233)
    mts_elnetcv = MultiOutputTsModel(
        elnetcv_model, train_x, train_y, test_x, test_y
    )
    mts_elnetcv.fit()
    pred_y_elnetcv = mts_elnetcv.predict()
    logger.info('Inference completed.')

    sc = SubmissionConvertor()

    averaging = WeightedAveraging(
        [pred_y_sarima.iloc[-5:], pred_y_elnetcv], [0.7, 0.3]
    )
    # averaging = WeightedAveraging(
    #     [pred_y_sarima.iloc[-5:], pred_y_elnetcv, pred_y_lgb], [0.7, 0.3, 0]
    # )
    pred_y_fusion = averaging.predict()
    pred_y_fusion = sc.np2df(pred_y_fusion).astype(int)


    """Weekday apply averaging."""
    final_df = final_df.astype(int)
    temp = np.array(pred_y_fusion[['reg_pv', 'reg_uv']], dtype='int')
    final_df.loc[:4, ['reg_pv', 'reg_uv']] = temp


    """Simple Averaging"""
    train_x, train_y, test_x = gen_train_test_weekend_cross(train)
    test_y = cfg.submit.sample
    test_y = sc.sub2df(test_y).iloc[:5]
    logger.info('Start training base model B...')
    lgb_model = lgb.LGBMRegressor(
        boosting_type='gbdt',
        num_leaves=42,
        reg_alpha=0,
        reg_lambda=0,
        max_depth=8,
        learning_rate=0.01,
        min_child_samples=5,
        n_estimators=2000,
        subsample=0.7,
        colsample_bytree=0.5,
        random_state=233,
        silent=True)
    mts_lgb = MultiOutputTsModel(lgb_model, train_x, train_y, test_x, test_y)
    mts_lgb.fit()
    pred_y_lgb = mts_lgb.predict().astype(int)
    logger.info('Inference completed.')

    temp = np.hstack([
        np.array(pred_y_lgb.reg_pv), final_df.reg_pv.values[-2:]
    ])
    temp = temp * .6 + final_df.reg_pv * .4
    final_df.reg_pv = temp.astype(int)

    logger.info('End!')
    return final_df, pred_y_lgb
