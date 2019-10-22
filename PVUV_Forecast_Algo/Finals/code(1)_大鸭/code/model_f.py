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


def model_f(train, model_df):
    sc = SubmissionConvertor()
    train = sc.sub2df(train)
    print('train', train)
    final_df = model_df.copy()
    logger.info('===STAGE E===')
    final_df.iloc[-2:] = (train.iloc[-9:-7].values + train.iloc[-16:-14].values)/2

    logger.info('Inference completed.')

    sc = SubmissionConvertor()

    final_df = final_df.astype(int)
    # temp = np.array(pred_y_sarima[['reg_pv', 'reg_uv']], dtype='int')
    # final_df.loc[:4, ['reg_pv', 'reg_uv']] = temp

    logger.info('End!')
    return final_df, final_df
