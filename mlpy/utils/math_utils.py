# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import math


def cal_euclidean_distance(x1, x2):
    return math.sqrt(np.sum(np.power(x1, 2) - np.power(x2, 2), axis=0))


