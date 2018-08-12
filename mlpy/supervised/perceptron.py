# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

from mlpy.utils.activation_functions import Sigmoid
from mlpy.utils.loss_functions import SquareLoss
from mlpy.utils.optimizers import SGD

class Perceptron():
    def __init__(self, iterations=1000, activation_function=Sigmoid, loss_function=SquareLoss, optimizer=SGD):
        return
