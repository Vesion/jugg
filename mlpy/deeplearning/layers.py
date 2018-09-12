# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from mlpy.utils.activation_functions import *


class _Layer_():
    '''
    The super abstract layer class.
    '''
    @property
    def name(self):
        return self.__class__.__name__

    def set_input_shape(self, input_shape):
        ''' Set the shape that the layer expects of the input in forward. '''
        self.input_shape = input_shape

    def forward(self, X, is_train):
        ''' Forward propagate signals. '''
        raise NotImplementedError()

    def backward(self, accum_grad):
        ''' Backward propagate the accumulated gradients.
        Receives gradients come from the next layer, produces gradients for previous layer. '''
        raise NotImplementedError()

    def output_shape(self):
        ''' The shape of output produced by forward propagation. '''
        raise NotImplementedError()

    def parameter_size(self):
        ''' The number of parameters (weights and bias) used by the layer. '''
        return 0


class Dense(_Layer_):
    '''
    Fully connected layer.
    '''
    def __init__(self, n_units, input_shape=None):
        '''
        n_units: n
          The number of neurons in the layer, also decide the output shape.
        input_shape: (m, )
          The expected input shape of the layer, specifying the number of features of the input.
          Must be set if it is the first layer in the network.
        '''
        self.n_units = n_units
        self.W = None
        self.w0 = None
        self.set_input_shape(input_shape)

    def initialize(self, optimizer):
        limit = 1 / np.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units)) # (m, n)
        self.w0 = np.zeros((1, self.n_units)) # (1, n)
        self.W_opt = optimizer()
        self.w0_opt = optimizer()

    def forward(self, X, is_train=True):
        # X: (k, m)
        self.X = X # (k, m)
        return X.dot(self.W) + self.w0 # (k, n)

    def output_shape(self):
        return (self.n_units, ) # (n, )

    def backward(self, accum_grad):
        # accum_grad: (k, n)
        old_W = self.W # (m, n)

        W_grad = self.X.T.dot(accum_grad) # (m, n)
        w0_grad = np.sum(accum_grad * 1, axis=0, keepdims=True) # (1, n)

        self.W = self.W_opt.update(self.W, W_grad) # (m, n)
        self.w0 = self.w0_opt.update(self.w0, w0_grad) # (1, n)

        return accum_grad.dot(old_W.T) # (k, m)

    def parameter_size(self):
        # m * n + n
        return np.prod(self.W.shape) + np.prod(self.w0.shape)


class Reshape(_Layer_):
    '''
    Reshape the input tensor to specified shape.
    '''
    def __init__(self, shape, input_shape=None):
        self.set_input_shape(input_shape)
        self.shape = shape

    def forward(self, X, is_train=True):
        self.prev_shape = X.shape
        return X.reshape((X.shape[0], ) + self.shape)

    def backward(self, accum_grad):
        return accum_grad.reshape(self.prev_shape)

    def output_shape(self):
        return self.shape


