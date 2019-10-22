"""
Performance Evaluation
"""

import pandas as pd
import numpy as np


def RMSPE(yhat, y):
    """RMSPE
    Root mean squared percentage error.

    :yhat:  predicted values.
    :y:     true values.
    """
    ret = ((yhat - y) / y)**2
    n = len(ret)
    ret = np.sum(ret)
    ret /= n
    ret = ret**0.5
    return ret


def eval_pred(pred, truth, verbose=True):
    """Evaluate in submission format.
    :pred:      predicted submission.
    :truth:     ground truth.
    :verbose:   detailed score.
    """
    rmspes = []
    for target in ['pv', 'uv']:
        for event in ['$pageview', 'reg_input_success']:
            rmspes.append(
                RMSPE(pred.loc[pred.event_type == event, target].astype(int).values,
                      truth.loc[pred.event_type == event, target].values))
    if verbose:
        print('===RMSPE===')
        print(rmspes, np.sum(rmspes))
    else:
        print(np.sum(rmspes))
