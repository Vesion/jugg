# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

from mlpy.utils.math_utils import cal_euclidean_distance


class KNearestNeighbors():
    def __init__(self, k=4):
        self.k = k

    def fit(self, features, label):
        # features: (k, m)
        # label: (k, 1)
        self.features = features
        self.label = label

    def _vote(self, neighbor_label):
        return np.bincount(neighbor_label.astype(int)).argmax()

    def predict(self, features):
        pred = np.zeros(features.shape[0])
        for i, sample in enumerate(features):
            # use the naive sort method to find the k nearest neighbors
            knn_idx = np.argsort([cal_euclidean_distance(sample, x) for x in self.features])[:self.k]
            knn = np.array([self.label[j][0] for j in knn_idx])
            pred[i] = self._vote(knn)
        return pred

KNN = KNearestNeighbors

