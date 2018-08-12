# Copyright @2018 xuxiang. All rights reserved.
#
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np

# https://en.wikipedia.org/wiki/Confusion_matrix

def accuracy_score(y_pred, y_label):
    # (TP+TN) / (TP+TN+FP+FN)
    accuracy = np.sum(y_pred == y_label, axis=0) / len(y_pred)
    return accuracy


def precision_score(y_pred, y_label):
    # TP / (TP+FP)
    TP = np.sum(y_label * (y_pred == y_label), axis=0)
    FP = np.sum((1-y_label) * (y_pred != y_label), axis=0)
    return TP / (TP + FP)


def recall_score(y_pred, y_label):
    # TP / (TP+FN)
    TP = np.sum(y_label * (y_pred == y_label), axis=0)
    FN = np.sum(y_label * (y_pred != y_label), axis=0)
    return TP / (TP + FN)


def F1_score(y_pred, y_label):
    # 2/F1 = 1/precision + 1/recall
    #   harmonic mean of precision and recall
    return 2 / (1/precision_score(y_pred, y_label) + 1/recall_score(y_pred, y_label))

