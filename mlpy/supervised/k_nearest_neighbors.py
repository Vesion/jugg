# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

from mlpy.utils.math_utils import cal_euclidean_distance


class KNearestNeighbors():
    '''
    Use brute-force way to find k nearest neighbor nodes.
    '''
    def __init__(self, k=4):
        self.k = k

    def fit(self, X, y):
        # X: (k, m)
        # y: (k, 1)
        self.X = X
        self.y = np.ravel(y) # convert to (k,)

    def _vote(self, neighbor_label):
        return np.bincount(neighbor_label.astype(int)).argmax()

    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for i, sample in enumerate(X):
            # use the naive sort method to find the k nearest neighbors
            knn_idx = np.argsort([cal_euclidean_distance(sample, x) for x in self.X])[:self.k]
            knn = np.array([self.y[j] for j in knn_idx])
            pred[i] = self._vote(knn)
        return pred

KNN = KNearestNeighbors

