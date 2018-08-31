# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import copy

from mlpy.utils.activation_functions import Tanh
from mlpy.utils.loss_functions import SquareLoss
from mlpy.utils.optimizers import SGD


class Perceptron():
    '''
    Apply non-linear activation function to linear output.
    '''
    def __init__(self, activation_function=Tanh, loss_function=SquareLoss, optimizer=SGD):
        self.activation_function = activation_function()
        self.loss_function = loss_function()
        self.optimizer_w = optimizer()
        self.optimizer_b = optimizer()

    def fit(self, features, label, iterations=100):
        # k: batch size
        # m: number of features in a sample
        # n: number of labels in a sample

        # features: (k, m)
        # label: (k, n)
        k = features.shape[1]
        n = label.shape[1]

        # weights: (m, n)
        # bias: (1, n)
        limit = 1 / np.sqrt(k)
        self.weights = np.random.uniform(-limit, limit, (k, n))
        self.bias = np.zeros((1, n))

        for i in range(iterations):
            linear_output = features.dot(self.weights) + self.bias # (k, n)
            pred = self.activation_function(linear_output) # (k, n)

            gradient = self.loss_function.gradient(pred, label) \
                     * self.activation_function.gradient(linear_output) # (k, n)
            gradient_w = features.T.dot(gradient) # (m, n)
            gradient_b = np.sum(gradient, axis=0, keepdims=True) # (1, n)

            self.weights = self.optimizer_w.update(self.weights, gradient_w)
            self.bias = self.optimizer_b.update(self.bias, gradient_b)

    def predict(self, features):
        linear_output = features.dot(self.weights) + self.bias
        pred = self.activation_function(linear_output)
        return pred

