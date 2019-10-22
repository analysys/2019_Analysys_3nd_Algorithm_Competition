import pandas as pd
import numpy as np
from models.base import ModelBase


class SimpleFilling(ModelBase):
    """
    Filling with the average with past values.
    """
    def __init__(self, memory=14, step=7):
        """
        :memory:    the time range to look back.
        :step:      step of averaging.
        """
        self.memory = memory
        self.step = step

    def fit(self, X, y):
        self.knowledge = y.iloc[-self.memory:].reset_index(drop=True)
        slices = []
        for i in range(0, self.memory, self.step):
            slices.append(
                self.knowledge.iloc[i:i + self.step].reset_index(drop=True)
            )
        self.history = pd.concat(slices, axis=1)

    def predict(self, X):
        return self.history.mean(axis=1)
