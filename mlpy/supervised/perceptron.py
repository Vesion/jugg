# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

from mlpy.utils.activation_functions import Tanh
from mlpy.utils.loss_functions import SquareLoss
from mlpy.utils.optimizers import SGD


class Perceptron():
    """
    The one layer neural network classifier.
    """
    def __init__(self, activation_function=Tanh, loss_function=SquareLoss, optimizer=SGD):
        self.activation_func = activation_function()
        self.loss_function = loss_function()
        self.optimizer = optimizer()

    def fit(self, X, y, iterations=100):
        # X: (k, m)
        # y: (k, n)
        m = X.shape[1]
        n = y.shape[1]

        # Initialize weights between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / np.sqrt(m)
        self.weights = np.random.uniform(-limit, limit, (m, n)) # (m, n)
        self.bias = np.zeros((1, n)) # (1, n)

        for i in range(iterations):
            linear_output = X.dot(self.weights) + self.bias # (k, n)
            pred = self.activation_func(linear_output) # (k, n)

            gradient = self.loss_function.gradient(y, pred) * self.activation_func.gradient(linear_output) # (k, n)
            gradient_w = X.T.dot(gradient) # (m, n)
            gradient_b = np.sum(gradient, axis=0, keepdims=True) # (1, n)

            self.weights = self.optimizer.update(self.weights, gradient_w) # (m, n)
            self.bias = self.optimizer.update(self.bias, gradient_b) # (1, n)

    def predict(self, X):
        pred = self.activation_func(X.dot(self.weights) + self.bias)
        return pred

