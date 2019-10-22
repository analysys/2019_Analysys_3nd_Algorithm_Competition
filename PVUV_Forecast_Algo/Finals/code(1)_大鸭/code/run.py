import pandas as pd
import numpy as np
import gc
from config import cfg, logger
from fileio.data_reader import read_files
from processing.preprocessing import nan_clean, DataPreparation, NoiseCancellation
from processing.feature import FeatureExtraction
from utils.submission import submit_export
from fileio.df_tools import SubmissionConvertor
from model_a import model_a
from model_b import model_b
from model_c import model_c
from model_d import model_d
from model_e import model_e
from model_f import model_f


def main():
    logger.info('==========Start==========')
    logger.info('Reading files...')
    # kpitrain, user, event = read_files(event=True)
    kpitrain, user = read_files()
    logger.info('File reading completed.')

    dp = DataPreparation()
    nc = NoiseCancellation()
    fe = FeatureExtraction()

    logger.info('Preparing data...')
    train = dp.kpitrain(kpitrain)
    train = nc.kpitrain(train)
    user = dp.user(user)

    # nan_clean(user, clean=True) # 清理高缺失
    # nan_clean(event, clean=True)
    gc.collect()
    logger.info('Data prepared.')
    logger.info('Extracting features...')
    train = fe.temporal_features(train)
    train = fe.lag_features(train)
    gc.collect()
    logger.info('Features extracted.')
    df_model_a, pred_a = model_a(train)
    df_model_e, pred_e = model_e(train, df_model_a)
    final_df = df_model_a * 0.46 + df_model_e * 0.5
    final_df, _ = model_f(kpitrain, final_df)
    print(final_df)

    sc = SubmissionConvertor()
    final_sub = sc.df2sub(final_df.astype(int))
    submit_export(final_sub, "BEST_A_E")
    logger.info('All done!')
    return pred_a, pred_e


if __name__ == '__main__':
    pred_a, pred_e = main()
