# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import copy

from mlpy.utils.activation_functions import Sigmoid
from mlpy.utils.loss_functions import SquareLoss
from mlpy.utils.optimizers import SGD


class Perceptron():
    def __init__(self, iterations=1000, activation_function=Sigmoid, loss_function=SquareLoss, optimizer=SGD):
        self.n_iterations = iterations
        self.activation_function = activation_function()
        self.loss_function = loss_function()
        self.optimizer_w = optimizer()
        self.optimizer_b = optimizer()

    def fit(self, features, label):
        # k: batch size
        # m: number of features in a sample
        # n: number of labels in a sample

        # features: [k, m]
        # label: [k, n]
        _, n_features = features.shape
        _, n_outputs = label.shape

        # weights: [m, n]
        # bias: [1, n]
        limit = 1 / np.sqrt(n_features)
        self.weights = np.random.uniform(-limit, limit, (n_features, n_outputs))
        self.bias = np.zeros((1, n_outputs))

        for i in range(self.n_iterations):
            linear_output = features.dot(self.weights) + self.bias # [k, n]
            pred = self.activation_function(linear_output) # [k, n]

            gradient = self.loss_function.gradient(pred, label) \
                     * self.activation_function.gradient(linear_output) # [k, n]
            gradient_w = features.T.dot(gradient) # [m, n]
            gradient_b = np.sum(gradient, axis=0, keepdims=True) # [1, n]

            self.weights = self.optimizer_w.update(self.weights, gradient_w)
            self.bias = self.optimizer_b.update(self.bias, gradient_b)

    def predict(self, features):
        linear_output = features.dot(self.weights) + self.bias
        pred = self.activation_function(linear_output)
        return pred

