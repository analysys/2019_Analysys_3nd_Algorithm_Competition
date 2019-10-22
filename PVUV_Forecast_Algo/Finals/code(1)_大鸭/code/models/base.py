from abc import ABCMeta, abstractmethod


class ModelBase(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
