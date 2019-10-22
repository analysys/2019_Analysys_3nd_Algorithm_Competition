import pandas as pd
import numpy as np
from config import cfg, logger
from fileio.data_reader import read_files
from processing.preprocessing import nan_clean, DataPreparation, NoiseCancellation
from processing.feature import FeatureExtraction
from processing.dataset_generator import gen_train_test_weekend
# from processing.dataset_generator import gen_train_test_weekend_cross
from models.statespace import StateSpaceModel
from models.simple_filling import SimpleFilling
from models.model import TsModel, MultiOutputTsModel
from utils.submission import submit_export
from fileio.df_tools import SubmissionConvertor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import ElasticNetCV
import lightgbm as lgb
from models.fusion import WeightedAveraging


def model_e(train, model_a_df):
    kpi_df = train.copy()
    sc = SubmissionConvertor()
    data = kpi_df.iloc[-21:].reset_index(drop=True)
    preds = []
    all_label = cfg.label.all
    for label in all_label:
        data_x = data[label].values
        sar = SARIMAX(data_x, order=(1, 1, 0), seasonal_order=(1, 0, 0, 7))
        sar_res = sar.fit(disp=0)
        preds.append(sar_res.forecast(7))
    sample = cfg.submit.sample
    sample_df = sc.sub2df(sample)
    for idx, p in enumerate(preds):
        sample_df.iloc[:, idx] = p


    final_df = model_a_df.copy()
    logger.info('===STAGE E===')
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
        sarima, None, train_y,
        None, test_y
    )
    # mts_sarima.fit(disp=0)
    # pred_y_sarima = mts_sarima.predict(
    #     start=0, end=len(train_x) + 4
    # )
    logger.info('Inference completed.')

    sc = SubmissionConvertor()

    final_df = final_df.astype(int)
    # temp = np.array(pred_y_sarima[['reg_pv', 'reg_uv']], dtype='int')
    # final_df.loc[:4, ['reg_pv', 'reg_uv']] = temp

    logger.info('End!')
    return sample_df, sample_df
