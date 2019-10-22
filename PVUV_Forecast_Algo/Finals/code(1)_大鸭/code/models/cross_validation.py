import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

INTERFACE_SUPPORT = ('sklearn', 'keras')

class CrossValidation(object):
    def __init__(self, model, random_seed=233, epochs=500,
                 verbose=2, patience=24, interface='sklearn', eval_func=None):
        self.model = model
        self.random_seed = random_seed
        self.epochs = epochs
        self.verbose = verbose
        self.patience = patience
        self.val_score = []
        self.cv_prediction = []
        self.interface = interface
        if interface not in INTERFACE_SUPPORT:
            raise NotImplementedError(
                'The interface {} is not supported yet.'.format(interface))
        self.eval_func = eval_func

    def fit(self, train_data, test_data, folds=5, **train_kwargs):
        """
        train_data : (train_x, train_y)
        test_data : (test_x, test_y)
        """
        self.folds = folds
        self.train_x = train_data[0]
        self.train_y = train_data[1]
        self.test_x = test_data[0]
        self.test_y = test_data[1]
        self.kf = KFold(n_splits=folds, random_state=self.random_seed)
        self._train_cv(self.kf, **train_kwargs)

    def _train_cv(self, kf, **train_kwargs):
        for idx, (trn_idx, val_idx) in enumerate(kf.split(self.train_x, )):
            trn_x = self.train_x.loc[self.train_x.index.isin(trn_idx), :].values
            trn_y = self.train_y.loc[self.train_y.index.isin(trn_idx), :].values
            val_x = self.train_x.loc[self.train_x.index.isin(val_idx), :].values
            val_y = self.train_y.loc[self.train_y.index.isin(val_idx), :].values
            print('Fold {}================='.format(idx + 1))
            # Fit models
            if self.interface == 'sklearn':
                self.model.fit(trn_x, trn_y.ravel(), **train_kwargs)
                score = self.eval_func(
                    val_y, self.model.predict(val_x)
                )
            elif self.interface == 'keras':
                self.model.fit(
                    trn_x, trn_y, epochs=self.epochs,
                    validation_data=(val_x, val_y),
                    callbacks=[
                        EarlyStopping(
                            monitor='val_loss', patience=self.patience
                        )
                    ],
                    **train_kwargs
                )
                score = self.model.evaluate(val_x, val_y)
            self.val_score.append(score)
            self.cv_prediction.append(self.model.predict(self.test_x))
        self.pred_test_y = self._get_pred_test_y()
        self.pred_train_y = self._get_pred_train_y()
        self.cv_score = np.mean(self.val_score)

    def _get_pred_test_y(self):
        pred_test = np.array(self.cv_prediction).mean(0)
        pred_test = pred_test.reshape(-1, 1)
        return pred_test

    def _get_pred_train_y(self):
        pred_train = self.model.predict(self.train_x)
        pred_train = np.array(pred_train).reshape(-1, 1)
        return pred_train

    def get_test_accuracy(self, eval_func):
        return eval_func(self.test_y, self.pred_test_y)

    def get_train_accuracy(self, eval_func):
        return eval_func(self.train_y, self.pred_train_y)

    @property
    def _model(self):
        return self.model

    @property
    def _train_x(self):
        return self.train_x

    @property
    def _train_y(self):
        return self.train_y

    @property
    def _test_x(self):
        return self.test_x

    @property
    def _test_y(self):
        return self.test_y

    @property
    def _cv_prediction(self):
        return self.cv_prediction

    @property
    def _cv_score(self):
        return self.cv_score

    @property
    def _val_score(self):
        return self.val_score

    @property
    def _pred_test_y(self):
        return self.pred_test_y
