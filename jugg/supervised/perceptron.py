# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np


class Perceptron():
    """
    Binary classifier, y = {-1, +1}
    A linear classifier combining a set of weights with the feature vector.
        f(x) = sign(w*x + b)
    """
    def __init__(self, learning_rate=0.5):
        self.learning_rate = learning_rate

    def _find_false_index(self):
        for i in range(len(self.X)):
            if self.y[i] * (self.X[i].dot(self.weights) + self.bias) <= 0:
                return i
        return None

    def fit(self, X, y):
        # X: (k, m)
        # y: (k, 1)
        self.X = X
        self.y = np.ravel(y) # convert to (k,)
        m = X.shape[1]

        # Initialize weights between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / np.sqrt(m)
        self.weights = np.random.uniform(-limit, limit, (m, 1)) # (m, 1)
        self.bias = 0

        false_index = self._find_false_index()
        while false_index != None:
            self.weights += self.learning_rate * self.y[false_index] * self.X[false_index].reshape(m, 1)
            self.bias += self.learning_rate * self.y[false_index]
            false_index = self._find_false_index()

    def predict(self, X):
        pred = np.sign(X.dot(self.weights) + self.bias)
        return np.ravel(pred)

