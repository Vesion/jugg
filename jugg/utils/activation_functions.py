# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

# https://en.wikipedia.org/wiki/Activation_function


class _Activation_Function_():
    @property
    def name(self):
        return self.__class__.__name__


class Linear(_Activation_Function_):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return self.alpha * x

    def gradient(self, x):
        return self.alpha


class Sigmoid(_Activation_Function_): # (0, 1)
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x): # y*(1-y)
        return self.__call__(x) * (1 - self.__call__(x))


class Softmax(_Activation_Function_): # (0, 1)
    def __call__(self, x):
        n_x = x - np.max(x, axis=-1, keepdims=True) # scale down to negative
        e_x = np.exp(n_x)
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class Tanh(_Activation_Function_): # (-1, 1)
    def __call__(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1
    
    def gradient(self, x): # 1-y**2
        return 1 - np.power(self.__call__(x), 2)


class ReLU(_Activation_Function_): # [0, +inf)
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)


class LeakyReLU(_Activation_Function_): # (-inf, +inf)
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)


class ELU(_Activation_Function_): # (-alpha, +inf)
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, x):
        return np.where(x >= 0, x, self.alpha * np.exp(x))


class Softplus(_Activation_Function_): # (0, +inf)
    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        return 1 / (1 + np.exp(-x))


class Gaussian(_Activation_Function_): # (0, 1]
    def __call__(self, x):
        return np.exp(-1 * np.power(x, 2))

    def gradient(self, x):
        return -2 * x * self.__call__(x)

