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


def main():
    logger.info('==========Start==========')
    logger.info('Reading files...')
    kpitrain, user, event = read_files(event=True)
    logger.info('File reading completed.')

    dp = DataPreparation()
    nc = NoiseCancellation()
    fe = FeatureExtraction()

    logger.info('Preparing data...')
    train = dp.kpitrain(kpitrain)
    train = nc.kpitrain(train)
    user = dp.user(user)

    nan_clean(user, clean=True) # 清理高缺失
    # nan_clean(event, clean=True)
    gc.collect()
    logger.info('Data prepared.')

    logger.info('Extracting features...')
    train = fe.temporal_features(train) # 时间特征
    # train = fe.filter_holiday(train)
    train = fe.lag_features(train) # 时序特征
    train = fe.userflow(train, user) # 用户量特征
    train = fe.osflow(train, event) # 浏览器特征
    train = fe.osflow_cross(train) # 时序交叉特征
    train = fe.create_window(train, cfg.feature.time_window)
    train = fe.window_stats(train, cfg.feature.time_window) # 滑窗统计
    train = fe.window_mean_diff(train) # 滑窗序列特征
    del event
    gc.collect()
    logger.info('Features extracted.')

    # Baseline，前两周平均
    df_model_a, pred_a = model_a(train)
    # 注册，SARIMA、线性、LGB
    df_model_b, pred_b = model_b(train, df_model_a)
    # 浏览人数，SARIMA、线性
    df_model_c, pred_c = model_c(train, df_model_b)
    # 浏览次数，线性
    df_model_d, pred_d = model_d(train, df_model_c)

    sc = SubmissionConvertor()
    final_sub = sc.df2sub(df_model_d)
    submit_export(final_sub, "BEST_A_B_C_D")

    logger.info('All done!')
    return pred_a, pred_b, pred_c, pred_d


if __name__ == '__main__':
    pred_a, pred_b, pred_c, pred_d = main()
