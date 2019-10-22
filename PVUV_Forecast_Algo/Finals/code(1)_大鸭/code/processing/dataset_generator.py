import pandas as pd
import numpy as np
from config import cfg


def gen_train_test(data):
    # data_x = data[cfg.feature.basic]
    data_x = data
    data_y = data[cfg.label.all]
    train_x = data_x[21:-7].reset_index(drop=True)
    train_y = data_y[21:-7].reset_index(drop=True)
    test_x = data_x[-7:].reset_index(drop=True)
    return train_x, train_y, test_x


def gen_train_test_weekend(data_):
    data = data_[data_.weekend == 0].reset_index(drop=True).drop('weekend', 1)
    # data_x = data[cfg.feature.weekend]
    data_x = data
    data_y = data[cfg.label.all]
    train_x = data_x[21:-5].reset_index(drop=True)
    train_y = data_y[21:-5].reset_index(drop=True)
    test_x = data_x[-5:].reset_index(drop=True)
    return train_x, train_y, test_x

# def gen_train_test_weekend_cross(data_):
#     data = data_[data_.weekend == 0].reset_index(drop=True).drop('weekend', 1)
#     data_x = data[cfg.feature.weekend + cfg.feature.cross]
#     data_y = data[cfg.label.all]
#     train_x = data_x[21:-5].reset_index(drop=True)
#     train_y = data_y[21:-5].reset_index(drop=True)
#     test_x = data_x[-5:].reset_index(drop=True)
#     return train_x, train_y, test_x
