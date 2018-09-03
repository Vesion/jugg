# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np


class DecisionStump():
    '''
    Used as the weak classifier, ensembled to make the strong classifier
    '''
    def __init__(self):
        # Determines if sample shall be classified as -1 or 1 given threshold
        self.polarity = 1

        self.feature_index = None

        self.threshold = None

        # the weight of this classifier in final model
        self.alpha = None


class AdaBoost():
    '''
    Adaptive Boosting uses a number of weak classifiers in ensemble to make a strong classifier.
    This implementation uses decision stumps, which is a one level Decision Tree. 
    '''
    def __init__(self, num_classifiers=5):
        self.n_clf = num_classifiers

    def fix(self, X, y):
        return


