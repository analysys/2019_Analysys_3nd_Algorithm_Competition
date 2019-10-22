import pandas as pd
import numpy as np
from config import cfg, logger
from processing.dataset_generator import gen_train_test
from models.statespace import StateSpaceModel
from models.simple_filling import SimpleFilling
from models.model import MultiOutputTsModel
from fileio.df_tools import SubmissionConvertor
from statsmodels.tsa.statespace.sarimax import SARIMAX


def model_a(train):
    logger.info('===STAGE A===')
    logger.info('Splitting dataset...')
    train_x, train_y, test_x = gen_train_test(train)

    sc = SubmissionConvertor()

    test_y = cfg.submit.sample
    test_y = sc.sub2df(test_y)

    # """Base Model A1 - SARIMAX"""
    # logger.info('Start training base model A1...')
    # sarima = StateSpaceModel(
    #     SARIMAX, order=(3, 2, 1), seasonal_order=(2, 1, 1, 7)
    # )
    # mts_sarima = MultiOutputTsModel(
    #     sarima, train_x[['userflow']], train_y,
    #     test_x[['userflow']], test_y
    # )
    # mts_sarima.fit(disp=0)
    # pred_y_sarima = mts_sarima.predict(
    #     start=0, end=len(train_x) + 6,
    #     exog=test_x[['userflow']]
    # )
    # pred_y_sarima = pred_y_sarima.iloc[-7:]
    # pred_sub_sarima = sc.df2sub(pred_y_sarima)
    # logger.info('Inference completed.')

    # # Final dataframe
    # final_df = pred_y_sarima.astype(int).reset_index(drop=True)

    """Base Model A2 - Last two weeks"""
    logger.info('Start training base model A2...')
    simple = SimpleFilling()
    mts_simple = MultiOutputTsModel(
        simple, train_x, train_y, test_x, test_y
    )
    mts_simple.fit()
    pred_y_simple = mts_simple.predict()
    logger.info('Inference completed.')

    """Weekend apply base model A2"""
    # final_df.loc[5:, ['reg_pv', 'reg_uv']] = pred_y_simple.loc[
    #     5:, ['reg_pv', 'reg_uv']].astype(int).values
    final_df = pred_y_simple.copy()
    final_sub = sc.df2sub(final_df)

    logger.info('End!')
    return final_df, pred_y_simple
