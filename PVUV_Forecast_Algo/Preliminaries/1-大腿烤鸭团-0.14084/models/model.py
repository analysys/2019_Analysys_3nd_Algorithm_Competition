import pandas as pd
import numpy as np
from copy import deepcopy
from models.cross_validation import CrossValidation
from models.base import ModelBase
from config import cfg, logger


class TsModel(ModelBase):
    """Time series model.
    Address the validation split and cv.
    :model:         predictive model.
    :train_x:       predictors.
    :train_y:       labels.
    :test_x:        predictors in the test set.
    :test_y:        labels in the test set.
    :val_size:      size of fixed validation set.
    """

    def __init__(self, model=None, train_x=None, train_y=None,
                 test_x=None, test_y=None, val_size=None):
        self.model = model
        self.train_x = train_x
        self.train_y = train_y
        self.test_x  = test_x
        if test_y is None:
            test_y = pd.DataFrame(
                np.zeros(test_x.shape[0], train_y.shape[1]))
        self.test_y  = test_y
        self.val_size = val_size

        if val_size is not None:
            self._split_validation()


    def _split_validation(self):
        """Split local validation set."""
        self.local_train_x = self.train_x.iloc[:-self.val_size]
        self.local_train_y = self.train_y.iloc[:-self.val_size]
        self.local_test_x  = self.test_x.iloc[-self.val_size:]
        self.local_test_y = self.test_y.iloc[-self.val_size:]

    def cv(self, eval_func='mse', folds=5, local=False):
        self.cv = CrossValidation(self.model, eval_func=eval_func)

        if local:
            cv_trn_x, cv_trn_y = self.local_train_x, self.local_train_y
            cv_tst_x, cv_tst_y = self.local_test_x,  self.local_test_y
        else:
            cv_trn_x, cv_trn_y = self.train_x, self.train_y
            cv_tst_x, cv_tst_y = self.test_x,  self.test_y

        self.cv.fit((cv_trn_x, cv_trn_y), (cv_tst_x, cv_tst_y), folds=folds)
        return self.cv


    def fit(self, X=None, y=None, local=False):
        X, y = self._use_local_xy(local, X, y)
        self.model.fit(X, y)

    def predict(self, X=None, local=False):
        X, _ = self._use_local_xy(local, _X=X, _y=None, test=True)
        return self.model.predict(X)

    def fit_predict(self, X=None, y=None, local=False):
        X, y = self._use_local_xy(local, X, y)
        self.fit(X, y)
        return self.predict(X)

    def _use_local_xy(self, local, _X, _y, test=False):
        if (_X is not None) and (_y is not None):
            X = _X
            y = _y
        elif local:
            if test:
                X = self.local_test_x
                y = self.local_test_y
            else:
                X = self.local_train_x
                y = self.local_train_y
        else:
            if test:
                X = self.test_x
                y = self.test_y
            else:
                X = self.train_x
                y = self.train_y
        return X, y


    @property
    def _model(self): return self.model

    @property
    def _train_x(self): return self.train_x

    @property
    def _train_y(self): return self.train_y

    @property
    def _test_x(self): return self.test_x

    @property
    def _test_y(self): return self.test_y

    @property
    def _cv(self): return self.cv


class MultiOutputTsModel(TsModel):
    def __init__(self, model=None, train_x=None, train_y=None,
                 test_x=None, test_y=None, val_size=None):
        super(MultiOutputTsModel, self).__init__(
            model, train_x, train_y, test_x, test_y, val_size
        )
        self.models = {
            'pg_pv':  deepcopy(model),
            'reg_pv': deepcopy(model),
            'pg_uv':  deepcopy(model),
            'reg_uv': deepcopy(model)
        }

    def cv(self, eval_func='mse', folds=5, local=False):
        self.cv = CrossValidation(self.model, eval_func=eval_func)
        self.cvs = {
            'pg_pv':  deepcopy(self.cv),
            'reg_pv': deepcopy(self.cv),
            'pg_uv':  deepcopy(self.cv),
            'reg_uv': deepcopy(self.cv)
        }

        if local:
            cv_trn_x, cv_trn_y = self.local_train_x, self.local_train_y
            cv_tst_x, cv_tst_y = self.local_test_x,  self.local_test_y
        else:
            cv_trn_x, cv_trn_y = self.train_x, self.train_y
            cv_tst_x, cv_tst_y = self.test_x,  self.test_y

        for i in range(4):
            self.cvs[cfg.label.mapper[i]].fit(
                (cv_trn_x, cv_trn_y.iloc[:, i]),
                (cv_tst_x, cv_tst_y.iloc[:, i]), folds=folds
            )
        return self.cvs


    def fit(self, X=None, y=None, local=False, **kwargs):
        X, y = self._use_local_xy(local, X, y)
        for i in range(4):
            idkey = cfg.label.mapper[i]
            self.models[idkey].fit(X, y.iloc[:, i], **kwargs)

    def predict(self, X=None, local=False, **kwargs):
        X, _ = self._use_local_xy(local, _X=X, _y=None, test=True)
        ret = {}
        for i in range(4):
            idkey = cfg.label.mapper[i]
            ret[idkey] = self.models[idkey].predict(X, **kwargs)
        return pd.DataFrame(ret)

    def fit_predict(self, X=None, y=None, local=False):
        X, y = self._use_local_xy(local, X, y)
        self.fit(X, y)
        return self.predict(X)
