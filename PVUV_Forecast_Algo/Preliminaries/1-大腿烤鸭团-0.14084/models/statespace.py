from models.base import ModelBase


class StateSpaceModel(ModelBase):
    """A sklearn wrapper for statsmodels statespace models.
    :model: model in statsmodels.tsa.statespace.
    """
    def __init__(self, model_instance, **params):
        self.model_instance = model_instance
        self.params = params

    def fit(self, X=None, y=None, **kwargs):
        self.model = self.model_instance(endog=y, exog=X, **self.params)
        self.model_res = self.model.fit(**kwargs)

    def predict(self, X, start=None, end=None, *args, **kwargs):
        return self.model_res.predict(start=start, end=end, *args, **kwargs)

    def fit_predict(self, X=0, y=None, start=None, end=None, fit_kwargs={}, predict_kwargs={}):
        self.fit(**fit_kwargs)
        return self.predict(start=start, end=end, **predict_kwargs)

    @property
    def _model(self): return self.model

    @property
    def _model_res(self): return self.model_res
