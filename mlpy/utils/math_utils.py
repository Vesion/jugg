# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import math


def cal_euclidean_distance(x1, x2):
    return math.sqrt(np.sum(np.power(x1, 2) - np.power(x2, 2), axis=0))


def cal_variance(X):
    # Calculate the variance of feature set X
    # X: (k, m)
    mean = X.mean(0)
    n_samples = X.shape[0]
    return (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))


def cal_std_dev(X):
    # Calculate the standard deviations of feature set X
    # X: (k, m)
    return np.sqrt(cal_variance(X))


def cal_label_entropy(Y):
    # Calculate the entropy of label set Y
    # Y: (k, 1).T
    unique_Y = np.unique(Y)
    entropy = 0
    for y in unique_Y:
        count = len(Y[Y == y])
        p = count / len(Y)
        entropy += -p * np.log2(p)
    return entropy


def cal_gaussian_distribution(x, mean, variance):
    eps = 1e-4 # to prevent division by zero
    coeff = 1.0 / math.sqrt(2.0 * math.pi * variance + eps)
    exponent = math.exp(-math.pow(x-mean, 2) / (2*variance + eps))
    return coeff * exponent

