"""
Model Results Fusion
"""
import pandas as pd
import numpy as np
from collections.abc import Iterable


def typed_property(name, expected_type):
    storage_name = '_' + name

    @property
    def prop(self):
        return getattr(self, storage_name)

    @prop.setter
    def prop(self, value):
        if not isinstance(value, expected_type):
            raise TypeError('{} must be a {}'.format(name, expected_type))
        setattr(self, storage_name, value)

    return prop


class WeightedAveraging():
    results = typed_property('results', Iterable)
    weights = typed_property('weights', Iterable)

    def __init__(self, results, weights=None):
        self.results = results
        if weights is None:
            self.weights = np.full(len(self.results), 1 / len(self.results))
        else:
            self.weights = weights
        if abs(np.sum(self.weights) - 1) > 1e-3:
            raise ValueError(
                'Sum of weights %.4f does not equal to 1.' %
                np.sum(self.weights)
            )

    def predict(self):
        self.weighted_results = []
        for res, w in zip(self.results, self.weights):
            self.weighted_results.append(res.values * w)
        # return np.sum(self.weighted_results, axis=0)
        temp = np.zeros_like(self.weighted_results[0]).ravel().tolist()
        for k in self.weighted_results:
            this = k.ravel().tolist()
            for idx,i in enumerate(this):
                temp[idx] = temp[idx] + i
            res = np.array(temp).reshape(
                self.weighted_results[0].shape[0],
                self.weighted_results[0].shape[1]
            )
        return res

    def __getitem__(self, idx):
        return self.results[idx], self.weights[idx]

    def __setitem__(self, idx, value):
        self.results[idx] = value[0]
        self.weights[idx] = value[1]



# class Blending(WeightedAveraging):
#     def __init__(self, truth, results, weights=None, blender=None):
#         super(Blending, self).__init__(results, weights)
#         self.truth = truth
#         self.blender = blender

#     def fit(self):
#         self.blender.fit(self.results, self.truth)

#     def predict(self, predictions):
#         self.blend_predictions = self.blender.predict(predictions)
#         return self.blend_predictions
