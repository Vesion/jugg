# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import cvxopt
from mlpy.utils.kernels import rbf_kernel

cvxopt.solvers.options['show_progress'] = False


class SupportVectorMachine():
    '''
    SVM with kernel method.
    Uses cvxopt to solve the convex quadratic programming.
    '''
    def __init__(self, C=1, kernel=rbf_kernel, power=4, gamma=None, coef=4):
        self.C = C
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.alpha = None # Lagrange Multiplier
        self.support_vectors = None
        self.support_vector_labels = None
        self.bias = None

    def fit(self, X, y):
        # X: (k, m)
        # y: (k, 1)
        k, m = np.shape(X)
        y = np.ravel(y) # convert to (k,)

        # Set gamma to 1/m by default
        if not self.gamma:
            self.gamma = 1 / m

        # Initialize kernel method with parameters
        self.kernel = self.kernel(
            power=self.power,
            gamma=self.gamma,
            coef=self.coef)

        # Calculate kernel matrix
        kernel_matrix = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])

        # Define the quadratic optimization problem
        # minimize 1/2 * x.T * P * x + q.T * x
        #     s.t. G * x <= h
        #          A * x == b
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(k) * -1)
        A = cvxopt.matrix(y, (1, k), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.C:
            G = cvxopt.matrix(np.identity(k) * -1)
            h = cvxopt.matrix(np.zeros(k))
        else:
            G_max = np.identity(k) * -1
            G_min = np.identity(k)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(k))
            h_min = cvxopt.matrix(np.ones(k) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Solve the quadratic optimization problem using cvxopt
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        alpha = np.ravel(minimization['x'])

        # Extract support vectors
        idx = alpha > 1e-7
        self.alpha = alpha[idx]
        self.support_vectors = X[idx]
        self.support_vector_labels = y[idx]

        # Calculate bias with first support vector
        self.bias = self.support_vector_labels[0]
        for i in range(len(self.alpha)):
            self.bias -= self.alpha[i] * self.support_vector_labels[
                i] * self.kernel(self.support_vectors[i], self.support_vectors[0])

    def predict(self, X):
        preds = []
        for sample in X:

            # pred = sum( alpha_i * y_i * Kernel(xi, x) ) + bias
            pred = 0
            for i in range(len(self.alpha)):
                pred += self.alpha[i] * self.support_vector_labels[i] \
                      * self.kernel(self.support_vectors[i], sample)
            pred += self.bias
            pred = np.sign(pred)

            preds.append(pred)
        return np.array(preds)

SVM = SupportVectorMachine

