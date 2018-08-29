# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from mlpy.utils.activation_functions import Sigmoid
from mlpy.utils.loss_functions import LogLoss
from mlpy.utils.optimizers import SGD


class LogisticRegression():
    def __init__(self, optimizer=SGD):
        self.activation_function = Sigmoid()
        self.loss_function = LogLoss()
        self.optimizer = optimizer()

    def fit(self, features, label, iterations=100):
        # k: batch size
        # m: num of features in a sample

        # feature: (k, m)
        # label: (k, 1)
        k = features.shape[1]

        # weight: (m, 1)
        limit = 1 / np.sqrt(k)
        self.weights = np.random.uniform(-limit, limit, (k,1))

        for i in range(iterations):
            linear_output = features.dot(self.weights) # (k, 1)
            pred = self.activation_function(linear_output) # (k, 1)

            gradient = self.loss_function.gradient(pred, label) \
                     * self.activation_function.gradient(linear_output) # (k, 1)
            gradient_w = features.T.dot(gradient) # (m, 1)

            self.weights = self.optimizer.update(self.weights, gradient_w) # (m, 1)

    def predict(self, features):
        linear_output = features.dot(self.weights)
        pred = self.activation_function(linear_output)
        return pred

LR = LogisticRegression;

