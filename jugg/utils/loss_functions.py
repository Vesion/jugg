# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

# https://en.wikipedia.org/wiki/Loss_functions_for_classification


class _Loss_Function_():
    @property
    def name(self):
        return self.__class__.__name__


class IndicatorLoss(_Loss_Function_): # 0-1 loss
    def __call__(self, y_pred, y_label):
        return 1 if np.all(y_pred == y_label) else 0

    def gradient(self, y_pred, y_label):
        raise NotImplementedError()

ZeroOneLoss = IndicatorLoss


class HingeLoss(_Loss_Function_):
    def __call__(self, y_pred, y_label):
        return max(0, np.max(1 - y_label * y_pred))

    def gradient(self, y_pred, y_label):
        raise NotImplementedError()


class SquareLoss(_Loss_Function_): # L2 loss
    def __call__(self, y_pred, y_label):
        return 0.5 * np.power((y_label - y_pred), 2)

    def gradient(self, y_pred, y_label):
        return -(y_label - y_pred)

L2Loss = SquareLoss


class AbsoluteLoss(_Loss_Function_): # L1 loss
    def __call__(self, y_pred, y_label):
        return np.abs(y_label - y_pred)

    def gradient(self, y_pred, y_label):
        raise NotImplementedError()

L1Loss = AbsoluteLoss


class CrossEntropy(_Loss_Function_): # log loss
    def __call__(self, y_pred, y_label):
        return -y_label * np.log(y_pred) - (1-y_label) * np.log(1-y_pred)

    def gradient(self, y_pred, y_label):
        return -y_label / y_pred + (1-y_label) / (1-y_pred)

LogLoss = CrossEntropy

