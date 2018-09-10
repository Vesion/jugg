# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from mlpy.utils.activation_functions import Sigmoid
from mlpy.utils.loss_functions import LogLoss
from mlpy.utils.optimizers import SGD


class LogisticRegression():
    '''
    Binary classifier, y = {0, 1}
    Log odds (logit) of prediction is the linear function to feature
    '''
    def __init__(self, optimizer=SGD):
        self.activation_function = Sigmoid()
        self.loss_function = LogLoss()
        self.optimizer = optimizer()

    def fit(self, X, y, iterations=100):
        # X: (k, m)
        # y: (k, 1)
        m = X.shape[1]

        # weight: (m, 1)
        limit = 1 / np.sqrt(m)
        self.weights = np.random.uniform(-limit, limit, (m,1))

        for i in range(iterations):
            linear_output = X.dot(self.weights) # (k, 1)
            pred = self.activation_function(linear_output) # (k, 1)

            gradient = self.loss_function.gradient(pred, y) \
                     * self.activation_function.gradient(linear_output) # (k, 1)
            gradient_w = X.T.dot(gradient) # (m, 1)

            self.weights = self.optimizer.update(self.weights, gradient_w) # (m, 1)

    def predict(self, X):
        linear_output = X.dot(self.weights)
        pred = self.activation_function(linear_output)
        return np.ravel(pred)

LR = LogisticRegression;

