import time
from config import cfg


def submit_export(df, prefix=None):
    """Export submission file.
    :df:        submission dataframe.
    :prefix:    file name prefix.
    """
    name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if prefix is not None:
        prefix = ''.join([prefix, '_'])
    name = ''.join(['submit_', prefix, name, '.csv'])
    df.pv = df.pv.astype(int)
    df.uv = df.uv.astype(int)
    df.to_csv(cfg.path.submission / name, index=None)
