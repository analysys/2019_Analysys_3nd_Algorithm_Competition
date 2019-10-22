import pandas as pd
import numpy as np
from config import cfg


class SubmissionConvertor():
    def __init__(self):
        pass

    @staticmethod
    def df2sub(df):
        predictions = [df[cfg.label.mapper[i]].values for i in range(4)]
        sub = cfg.submit.sample
        sub.loc[sub.event_type == cfg.label.pg, cfg.label.pv]  = predictions[0]
        sub.loc[sub.event_type == cfg.label.reg, cfg.label.pv] = predictions[1]
        sub.loc[sub.event_type == cfg.label.pg, cfg.label.uv]  = predictions[2]
        sub.loc[sub.event_type == cfg.label.reg, cfg.label.uv] = predictions[3]
        return sub

    @staticmethod
    def sub2df(sub):
        predictions = []
        predictions.append(sub.loc[sub.event_type == cfg.label.pg, cfg.label.pv].values)
        predictions.append(sub.loc[sub.event_type == cfg.label.reg, cfg.label.pv].values)
        predictions.append(sub.loc[sub.event_type == cfg.label.pg, cfg.label.uv].values)
        predictions.append(sub.loc[sub.event_type == cfg.label.reg, cfg.label.uv].values)
        df = pd.DataFrame({
            cfg.label.mapper[0]: predictions[0],
            cfg.label.mapper[1]: predictions[1],
            cfg.label.mapper[2]: predictions[2],
            cfg.label.mapper[3]: predictions[3]
        })
        return df

    @staticmethod
    def np2df(nparray):
        predictions = [nparray[:, i] for i in range(4)]
        df = pd.DataFrame({
            cfg.label.mapper[0]: predictions[0],
            cfg.label.mapper[1]: predictions[1],
            cfg.label.mapper[2]: predictions[2],
            cfg.label.mapper[3]: predictions[3]
        })
        return df
