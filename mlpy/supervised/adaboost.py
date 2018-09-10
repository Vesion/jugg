# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np


class DecisionStump():
    '''
    Binary classifier, y = {-1, +1}
    Used as the weak classifier, ensembled to make the strong classifier
    '''
    def __init__(self):
        # determines if sample shall be classified as -1 or 1 given threshold
        self.polarity = 1

        # the feature value used to make classification
        self.threshold = None
        # the index of the threshold feature
        self.feature_index = None

        # the weight of this classifier in final model
        self.alpha = None


class AdaBoost():
    '''
    Adaptive Boosting uses a number of weak classifiers in ensemble to make a strong classifier.
    This implementation uses decision stumps, which is a one level Decision Tree. 
    '''
    def __init__(self, num_classifiers=5):
        self.n_clfs = num_classifiers

    def fit(self, X, y):
        # X: (k, m)
        # y: (k, 1)
        k, m = X.shape
        y = np.ravel(y) # convert to (k,)

        w = np.full(k, (1 / k))
        self.clfs = []
        
        for _ in range(self.n_clfs):
            clf = DecisionStump()

            # find the minimum error and corresponding threshold to make classification
            min_error = float('inf')
            for feature_i in range(m):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                feature_uniques = np.unique(feature_values)
                for threshold in feature_uniques:
                    polarity = 1
                    pred = np.ones(y.shape)
                    pred[X[:, feature_i] < threshold] = -1

                    # error = sum of weights of misclassified samples
                    error = np.sum(w[pred != y])

                    # if error is over 0.5, we flip the polarity so that samples that
                    # were classified as -1 are classified as 1, and vice versa
                    if error > 0.5:
                        error = 1 - error
                        polarity = -1

                    if error < min_error:
                        clf.polarity = polarity
                        clf.feature_index = feature_i
                        clf.threshold = threshold
                        min_error = error

            # alpha is used to update sample weights
            # and also the weight of this classifier
            # classifier with smaller error gets larger alpha
            clf.alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-6))

            # predict 1 for sample values below threshold, -1 otherwise
            preds = np.ones(y.shape)
            negative_id = (clf.polarity * X[:, clf.feature_index]) < (clf.polarity * clf.threshold)
            preds[negative_id] = -1

            # calculate new weights based alpha, label and predictions
            w *= np.exp(-clf.alpha * y * preds)
            # normlalize to one
            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X):
        k = X.shape[0]
        y_preds = np.zeros((k, 1))
        for clf in self.clfs:
            # predict 1 for sample values below threshold, -1 otherwise
            preds = np.ones(y_preds.shape)
            negative_id = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            preds[negative_id] = -1

            # add predictions weighted by alpha
            y_preds += clf.alpha * preds
        y_preds = np.sign(y_preds)
        return np.ravel(y_preds)

