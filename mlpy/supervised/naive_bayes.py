# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from mlpy.utils.math_utils import cal_gaussian_distribution


class NaiveBayes():
    '''
    Multi-class classifier
    Bayes rule: P(Yi|X) = P(X|Yi)P(Yi) / P(X)
             or Posterior = Likelihood * Prior / Scaling Factor
    Classifies the sample as the class that results in the largest P(Y|X) (posterior)
    P(Yi|X) - Posterior, the probability that the sample X is class Yi,
              given the feature X being distributed according to prior distribution Y
    P(X|Yi) - Likelihood of feature X given class Yi
              use Gaussian Distribution here
    P(Yi)   - Prior, distribution of classes Y
    P(X)    - Scales the posterior to make it a proper probability distribution,
              this term is ignored in this implementation since it doesn't affect
              which class distribution the sample is most likely to belong to.
    '''
    def fit(self, X, y):
        # X: (k, m)
        # y: (k, 1)
        self.X = X
        self.y = np.ravel(y) # convert to (k,)
        self.classes = np.unique(y)
        self.params = []

        for i, c in enumerate(self.classes):
            X_c = X[np.where(y == c)]
            self.params.append([])
            for feature in X_c.T:
                self.params[i].append({'mean': feature.mean(), 'var': feature.var()})
        

    def _cal_likelihood(self, x, mean, var):
        # use Gaussian Distribution (Normal Distribution)
        return cal_gaussian_distribution(x, mean, var)

    def _cal_prior(self, c):
        frequency = np.mean(self.y == c)
        return frequency

    def _classify(self, x):
        posteriors = np.zeros(self.classes.shape)

        for i, c in enumerate(self.classes):
            prior = self._cal_prior(c)
            posteriors[i] = prior

            for feature, params in zip(x, self.params[i]):
                likelihood = self._cal_likelihood(feature, params['mean'], params['var'])
                posteriors[i] *= likelihood

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        preds = [self._classify(x) for x in X]
        return preds

