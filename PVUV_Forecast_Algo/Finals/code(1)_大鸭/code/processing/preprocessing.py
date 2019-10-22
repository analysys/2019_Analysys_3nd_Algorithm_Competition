import pandas as pd
import numpy as np
import gc
from config import cfg


def nan_clean(df, clean=False, miss=10, verbose=False):
    """Clean NaN values.
    :df:        input dataframe.
    :clean:     delete columns with too many nans.
    :miss:      threshold to deleta a column.
    :verbose:   print process.
    """
    for c in df.columns:
        missing_rate = len(df[df[c].isna()]) / len(df) * 100
        if verbose:
            print("%s:\t%.1f" % (c, missing_rate), end='')
            if clean and missing_rate > miss:
                del df[c]
                print('\tCLEAN!')
            else:
                print('')
        else:
            if clean and missing_rate > miss:
                del df[c]


class DataPreparation():
    """
    Prepare the dataframe for later anaylsis.
    """
    def __init__(self):
        pass

    @staticmethod
    def kpitrain(train_):
        train = pd.concat([
            train_.pivot('date', 'event_type',
                        'pv').reset_index().rename(columns={
                            '$pageview': 'pg_pv',
                            'reg_input_success': 'reg_pv'
                        }),
            train_.pivot('date', 'event_type',
                        'uv').reset_index().rename(columns={
                            '$pageview': 'pg_uv',
                            'reg_input_success': 'reg_uv'
                        })[['pg_uv', 'reg_uv']]
        ], axis=1).reset_index(drop=True)

        submit_sample = cfg.submit.sample
        train_date_expand = pd.DataFrame(
            pd.to_datetime(submit_sample.date, format='%Y%m%d'),
            columns=['date']).drop_duplicates()
        train = train.merge(train_date_expand, 'outer').reset_index(drop=True)
        gc.collect()
        return train

    @staticmethod
    def user(user_):
        user = user_.copy()
        user['xwhen_date'] = user.xwhen.dt.date

        return user


class NoiseCancellation():
    """
    Modify some noisy values manually.
    """
    def __init__(self):
        pass

    @staticmethod
    def kpitrain(train_):
        train = train_.copy()
        train.iloc[74, 3] = 1875
        train.iloc[75, 3] = 1800

        train.loc[(train.reg_pv / train.reg_uv) > 2, 'reg_pv'] = train.loc[
            (train.reg_pv / train.reg_uv) > 2, 'reg_uv'] * (
                train.reg_pv / train.reg_uv).median()

        return train
