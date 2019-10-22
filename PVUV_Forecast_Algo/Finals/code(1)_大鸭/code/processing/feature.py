import pandas as pd
import numpy as np
from config import cfg


class FeatureExtraction():
    """
    Feature extraction.
    """
    def __init__(self):
        pass

    @staticmethod
    def temporal_features(train_):
        train = train_.copy()
        train['dayofweek'] = train.date.dt.dayofweek
        train['weekend'] = (train.dayofweek > 4).astype(int)
        return train

    @staticmethod
    def filter_holiday(train_):
        train = train_.copy()
        filters = (
            train.date.between(pd.datetime(2019, 4, 29), pd.datetime(2019, 5, 5)) |
            train.date.between(pd.datetime(2019, 4, 8), pd.datetime(2019, 4, 14)) |
            train.date.between(pd.datetime(2019, 1, 28), pd.datetime(2019, 2, 10)) |
            train.date.between(pd.datetime(2018, 12, 3), pd.datetime(2018, 12, 9))
        )
        return train[~filter_holiday].reset_index(drop=True)

    @staticmethod
    def lag_features(train_):
        train = train_.copy()

        train['pg_pv_lastweek'] = train.pg_pv.shift(7)
        train['reg_pv_lastweek'] = train.reg_pv.shift(7)
        train['pg_uv_lastweek'] = train.pg_uv.shift(7)
        train['reg_uv_lastweek'] = train.reg_uv.shift(7)

        train['pg_pv_last2week'] = train.pg_pv.shift(14)
        train['reg_pv_last2week'] = train.reg_pv.shift(14)
        train['pg_uv_last2week'] = train.pg_uv.shift(14)
        train['reg_uv_last2week'] = train.reg_uv.shift(14)

        train['pg_pv_diff_lastweek'] = train.pg_pv.diff(7).shift(7)
        train['reg_pv_diff_lastweek'] = train.reg_pv.diff(7).shift(7)
        train['pg_uv_diff_lastweek'] = train.pg_uv.diff(7).shift(7)
        train['reg_uv_diff_lastweek'] = train.reg_uv.diff(7).shift(7)

        train['pg_pv_diff_last2week'] = train.pg_pv.diff(14).shift(7)
        train['reg_pv_diff_last2week'] = train.reg_pv.diff(14).shift(7)
        train['pg_uv_diff_last2week'] = train.pg_uv.diff(14).shift(7)
        train['reg_uv_diff_last2week'] = train.reg_uv.diff(14).shift(7)

        train['pg_pv_sum_last2week'] = train.pg_pv_lastweek + train.pg_pv_last2week
        train['reg_pv_sum_last2week'] = train.reg_pv_lastweek + train.reg_pv_last2week
        train['pg_uv_sum_last2week'] = train.pg_uv_lastweek + train.pg_uv_last2week
        train['reg_uv_sum_last2week'] = train.reg_uv_lastweek + train.reg_uv_last2week

        return train

    # @staticmethod
    # def userflow(train_, user):
    #     userflow = user.groupby('xwhen_date').size().reset_index().rename(
    #         columns={
    #             'xwhen_date': 'date',
    #             0: 'userflow'
    #         }
    #     )
    #     userflow.date = pd.to_datetime(userflow.date)
    #     train = train_.merge(userflow)
    #     train['userflow_group'] = train.userflow // 5
    #
    #     return train

    # @staticmethod
    # def osflow(train_, event):
    #     train = train_.copy()
    #     for o in event['$os'].unique():
    #         o_ = '_'.join(['os', o])
    #         o_g = '_'.join([o_, 'group'])
    #         event_os = event.loc[event['$os'] == o]
    #         o_flow = event_os.groupby('date').size().reset_index(name=o_)
    #         train = train.merge(
    #             o_flow, 'left', on='date'
    #         )
    #         train[o_] = train[o_].fillna(0)
    #         train[o_g] = train[o_] // 5
    #     return train

    # @staticmethod
    # def osflow_cross(train_):
    #     train = train_.copy()
    #     for y in cfg.label.all:
    #         for o in cfg.event.os + ['userflow']:
    #             if o == 'userflow':
    #                 temp = train
    #                 o_g = 'userflow_group'
    #                 o_g_y = '_'.join([o_g, y])
    #             else:
    #                 temp = train.iloc[:-7]
    #                 o_g = '_'.join(['os', o, 'group'])
    #                 o_g_y = '_'.join(['os', o, y])
    #             o_mean = temp.groupby(o_g)[y].mean().reset_index(name=o_g_y)
    #             train = train.merge(o_mean, 'left', on=o_g)
    #     return train

    @staticmethod
    def create_window(train_, window=14):
        df = train_.copy()
        for w in range(window):
            shift_df = df[cfg.label.window + cfg.label.all].shift(7 + w).copy()
            shift_df.columns = ['_lag_'.join([c, str(w)]) for c in shift_df.columns]
            df = pd.concat([df, shift_df], axis=1)
        return df

    @staticmethod
    def window_stats(train_, window=14, step=7):
        df = train_.copy()
        # for y in cfg.label.window:
        #     for window_start in range(0, window, step):
        #         lag_df_y = df[[
        #             '_lag_'.join([y, str(w)])
        #             for w in range(window_start, window_start + step)
        #         ]]
        #         stat_lag_df_y = pd.DataFrame({
        #             '_'.join([y, 'lag', str(window_start), 'mean']): np.mean(lag_df_y, axis=1),
        #             '_'.join([y, 'lag', str(window_start), 'std']): np.std(lag_df_y, axis=1),
        #         })
        #         df = pd.concat([df, stat_lag_df_y], axis=1)
        for y in cfg.label.all:
            for window_start in range(0, window, step):
                lag_df_y = df[[
                    '_lag_'.join([y, str(w)])
                    for w in range(window_start, window_start + step)
                ]]
                stat_lag_df_y = pd.DataFrame({
                    '_'.join([y, 'lag', str(window_start), 'mean']): np.mean(lag_df_y, axis=1),
                    '_'.join([y, 'lag', str(window_start), 'min']): np.min(lag_df_y, axis=1),
                    '_'.join([y, 'lag', str(window_start), 'max']): np.max(lag_df_y, axis=1),
                    '_'.join([y, 'lag', str(window_start), 'std']): np.std(lag_df_y, axis=1),
                    # '_'.join([y, 'lag', str(window_start), 'ptp']): np.ptp(lag_df_y, axis=1),
                })
                df = pd.concat([df, stat_lag_df_y], axis=1)
        return df

    @staticmethod
    def window_mean_diff(train_):
        train = train_.copy()
        for y in cfg.label.all:
            y_lag_mean_diff = '_'.join([y, 'lag', 'mean', 'diff'])
            y_lag0 = '_'.join([y, 'lag_0_mean'])
            y_lag7 = '_'.join([y, 'lag_7_mean'])
            train[y_lag_mean_diff] = train[y_lag0] - train[y_lag7]
        return train
