# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

class _Optimizer_():
    @property
    def name(self):
        return self.__class__.__name__


class StochasticGradientDescent(_Optimizer_):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None

    def update(self, weights, gradients):
        if self.v is None:
            self.v = np.zeros(weights.shape)
        self.v = self.momentum * self.v + self.learning_rate * gradients
        return weights - self.v

SGD = StochasticGradientDescent


class NesterovAcceleratedGradient(_Optimizer_):
    def __init__(self, learning_rate=0.01, momentum=0.2):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None

    def update(self, weights, gradients):
        if self.v is None:
            self.v = np.zeros(weights.shape)

        self.v = self.momentum * self.v + \
                 self.learning_rate * (gradient - self.momentum * self.v)
        return weights - self.v

NAGD = NesterovAcceleratedGradient


class Adagrad(_Optimizer_):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.G = None
        self.eps = 1e-8

    def update(self, weights, gradients):
        if self.G is None:
            self.G = np.zeros(weights.shape)
        self.G += np.power(weights, 2)
        return weights - self.learning_rate * gradients / np.sqrt(self.G + self.eps) 


class Adadelta(_Optimizer_):
    def __init__(self, rho=0.95, eps=1e-6):
        self.v = None
        self.s_v = None
        self.s_g = None
        self.eps = eps
        self.rho = rho

    def update(self, weights, gradients):
        if self.v is None:
            self.v = np.zeros(np.shape(weights))
            self.s_v = np.zeros(np.shape(weights))
            self.s_g = np.zeros(np.shape(gradients))

        self.s_g = self.rho * self.s_g + (1 - self.rho) * np.power(gradients, 2)
        RMS_delta_w = np.sqrt(self.s_v + self.eps)
        RMS_grad = np.sqrt(self.s_g + self.eps)

        adaptive_lr = RMS_delta_w / RMS_grad # Adaptive learning rate
        self.v = adaptive_lr * gradients
        self.s_v = self.rho * self.s_v + (1 - self.rho) * np.power(self.v, 2)

        return weights - self.v


class RMSprop():
    def __init__(self, learning_rate=0.01, rho=0.9):
        self.learning_rate = learning_rate
        self.Eg = None # average of the square gradients at weights
        self.eps = 1e-8
        self.rho = rho

    def update(self, weights, gradients):
        if self.Eg is None:
            self.Eg = np.zeros(np.shape(gradients))

        self.Eg = self.rho * self.Eg + (1 - self.rho) * np.power(gradients, 2)

        # Divide the learning rate for a weight by a running average of the magnitudes of recent gradients for that weight
        return weights - self.learning_rate *  gradients / np.sqrt(self.Eg + self.eps)


class Adam():
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.m = None
        self.v = None
        # Decay rates
        self.b1 = b1
        self.b2 = b2

    def update(self, weights, gradients):
        if self.m is None:
            self.m = np.zeros(np.shape(gradients))
            self.v = np.zeros(np.shape(gradients))
        
        self.m = self.b1 * self.m + (1 - self.b1) * gradients
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(gradients, 2)

        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        return weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

