"""
Read data files.
"""
import pandas as pd
import numpy as np
from config import cfg


def ms_date_parser(x):
    """Parse date from ms to pd.datetime."""
    return pd.to_datetime(x, unit='ms')

def read_files(kpitrain=True, user=True, event=False) -> list:
    files = []
    if kpitrain:
        kpitrain = pd.read_csv(cfg.data.kpitrain, parse_dates=['date'])
        files.append(kpitrain)

    if user:
        user = pd.read_csv(cfg.data.user, parse_dates=['xwhen'], date_parser=ms_date_parser)
        user['date'] = user['xwhen'].dt.date
        files.append(user)

    if event:
        eventdetail = pd.read_csv(cfg.data.event, parse_dates=['time'])
        eventdetail['$os'] = eventdetail['$os'].map(cfg.event.os_mapper).fillna('others')
        eventdetail = eventdetail.rename(columns={
            'time': 'date'
        })
        files.append(eventdetail)

    return files
