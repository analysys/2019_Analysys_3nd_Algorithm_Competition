import pandas as pd
import numpy as np
import logging
from pathlib import Path
from addict import Dict


"""
Configurations
"""
_C = Dict()

_C.path.root        = Path('./data/')
_C.path.submission  = _C.path.root
_C.path.data        = _C.path.root

_C.data.kpitrain    = _C.path.data / 'kpi_train.csv'
_C.data.user        = _C.path.data / 'user.csv'
_C.data.event       = _C.path.data / 'event_detail.csv'

_C.submit.sample    = pd.read_csv(_C.path.submission / 'submit_sample.csv')

_C.event.os_mapper  = {
    'Windows': 'windows',
    'Mac OS X': 'mac',
    'Windows NT': 'windows',
    'Android': 'android',
    'iOS': 'ios',
    'others': 'others',
    'Ubuntu': 'others',
    'Microsoft Cloud': 'windows',
    'Cloud': 'others',
    'Unknown': 'others',
    'Fedora': 'others',
    'Chrome OS': 'others',
    'Windowsx': 'windows',
    'Hacker': 'others',
    'Google': 'others',
    'Tizen': 'others',
    'Series60': 'others',
    np.nan: 'others'
}
_C.event.os = ['windows', 'mac', 'android', 'ios', 'others']

_C.feature.time_window  = 14
_C.feature.basic = [
    'dayofweek', 'weekend', 'userflow', 'pg_pv_lastweek', 'reg_pv_lastweek',
    'pg_uv_lastweek', 'reg_uv_lastweek', 'pg_pv_diff_lastweek',
    'reg_pv_diff_lastweek', 'pg_uv_diff_lastweek', 'reg_uv_diff_lastweek',
    'pg_pv_last2week', 'reg_pv_last2week', 'pg_uv_last2week',
    'reg_uv_last2week', 'pg_pv_diff_last2week', 'reg_pv_diff_last2week',
    'pg_uv_diff_last2week', 'reg_uv_diff_last2week', 'pg_pv_sum_last2week',
    'reg_pv_sum_last2week', 'pg_uv_sum_last2week', 'reg_uv_sum_last2week',
    'pg_pv_lag_0_mean', 'pg_pv_lag_0_min', 'pg_pv_lag_0_max',
    'pg_pv_lag_0_std', 'pg_pv_lag_0_ptp', 'pg_pv_lag_7_mean',
    'pg_pv_lag_7_min', 'pg_pv_lag_7_max', 'pg_pv_lag_7_std', 'pg_pv_lag_7_ptp',
    'reg_pv_lag_0_mean', 'reg_pv_lag_0_min', 'reg_pv_lag_0_max',
    'reg_pv_lag_0_std', 'reg_pv_lag_0_ptp', 'reg_pv_lag_7_mean',
    'reg_pv_lag_7_min', 'reg_pv_lag_7_max', 'reg_pv_lag_7_std',
    'reg_pv_lag_7_ptp', 'pg_uv_lag_0_mean', 'pg_uv_lag_0_min',
    'pg_uv_lag_0_max', 'pg_uv_lag_0_std', 'pg_uv_lag_0_ptp',
    'pg_uv_lag_7_mean', 'pg_uv_lag_7_min', 'pg_uv_lag_7_max',
    'pg_uv_lag_7_std', 'pg_uv_lag_7_ptp', 'reg_uv_lag_0_mean',
    'reg_uv_lag_0_min', 'reg_uv_lag_0_max', 'reg_uv_lag_0_std',
    'reg_uv_lag_0_ptp', 'reg_uv_lag_7_mean', 'reg_uv_lag_7_min',
    'reg_uv_lag_7_max', 'reg_uv_lag_7_std', 'reg_uv_lag_7_ptp'
]
_C.feature.weekend = [
    'dayofweek', 'userflow', 'pg_pv_lastweek', 'reg_pv_lastweek',
    'pg_uv_lastweek', 'reg_uv_lastweek', 'pg_pv_diff_lastweek',
    'reg_pv_diff_lastweek', 'pg_uv_diff_lastweek', 'reg_uv_diff_lastweek',
    'pg_pv_last2week', 'reg_pv_last2week', 'pg_uv_last2week',
    'reg_uv_last2week', 'pg_pv_diff_last2week', 'reg_pv_diff_last2week',
    'pg_uv_diff_last2week', 'reg_uv_diff_last2week', 'pg_pv_sum_last2week',
    'reg_pv_sum_last2week', 'pg_uv_sum_last2week', 'reg_uv_sum_last2week',
    'pg_pv_lag_0_mean', 'pg_pv_lag_0_min', 'pg_pv_lag_0_max',
    'pg_pv_lag_0_std', 'pg_pv_lag_0_ptp', 'pg_pv_lag_7_mean',
    'pg_pv_lag_7_min', 'pg_pv_lag_7_max', 'pg_pv_lag_7_std', 'pg_pv_lag_7_ptp',
    'reg_pv_lag_0_mean', 'reg_pv_lag_0_min', 'reg_pv_lag_0_max',
    'reg_pv_lag_0_std', 'reg_pv_lag_0_ptp', 'reg_pv_lag_7_mean',
    'reg_pv_lag_7_min', 'reg_pv_lag_7_max', 'reg_pv_lag_7_std',
    'reg_pv_lag_7_ptp', 'pg_uv_lag_0_mean', 'pg_uv_lag_0_min',
    'pg_uv_lag_0_max', 'pg_uv_lag_0_std', 'pg_uv_lag_0_ptp',
    'pg_uv_lag_7_mean', 'pg_uv_lag_7_min', 'pg_uv_lag_7_max',
    'pg_uv_lag_7_std', 'pg_uv_lag_7_ptp', 'reg_uv_lag_0_mean',
    'reg_uv_lag_0_min', 'reg_uv_lag_0_max', 'reg_uv_lag_0_std',
    'reg_uv_lag_0_ptp', 'reg_uv_lag_7_mean', 'reg_uv_lag_7_min',
    'reg_uv_lag_7_max', 'reg_uv_lag_7_std', 'reg_uv_lag_7_ptp'
]
_C.feature.cross = [
    'userflow_group_pg_pv_lag_0_mean',
    'userflow_group_pg_pv_lag_0_std',
    'userflow_group_pg_pv_lag_7_mean',
    'userflow_group_pg_pv_lag_7_std',
    'userflow_group_reg_pv_lag_0_mean',
    'userflow_group_reg_pv_lag_0_std',
    'userflow_group_reg_pv_lag_7_mean',
    'userflow_group_reg_pv_lag_7_std',
    'userflow_group_pg_uv_lag_0_mean',
    'userflow_group_pg_uv_lag_0_std',
    'userflow_group_pg_uv_lag_7_mean',
    'userflow_group_pg_uv_lag_7_std',
    'userflow_group_reg_uv_lag_0_mean',
    'userflow_group_reg_uv_lag_0_std',
    'userflow_group_reg_uv_lag_7_mean',
    'userflow_group_reg_uv_lag_7_std'
]

_C.label.mapper = {
    0: 'pg_pv',
    1: 'reg_pv',
    2: 'pg_uv',
    3: 'reg_uv'
}
_C.label.all = ['pg_pv', 'reg_pv', 'pg_uv', 'reg_uv']
_C.label.window = [
    '_'.join(['os', o, y])
    for y in _C.label.all
    for o in _C.event.os
] + [
    '_'.join(['userflow_group', y])
    for y in _C.label.all
] + [
    '_'.join(['os', o])
    for o in _C.event.os
]
_C.label.pg = '$pageview'
_C.label.reg = 'reg_input_success'
_C.label.pv = 'pv'
_C.label.uv = 'uv'


cfg = _C


"""
Logging
"""
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
