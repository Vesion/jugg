import numpy as np

# ## utils
#
# from jugg.utils.activation_functions import *
# from jugg.utils.loss_functions import *
# from jugg.utils.metrics import *
# from jugg.utils.optimizers import *

# a = np.array([1,0,1,1], dtype=float)
# b = np.array([1,0,0,1], dtype=float)

# print accuracy_score(a, b)
# print precision_score(a, b)
# print recall_score(a, b)
# print F1_score(a, b)

# p = np.array([0.2,0.8], dtype=float)
# y = np.array([0,1], dtype=float)
# l = CrossEntropy()
# l = SquareLoss()
# l = L1Loss()
# print l(p, y)


# peceptron
from jugg.supervised.perceptron import Perceptron

x = np.array([
    [3,3],
    [4,3],
    [1,1],
], dtype=float)

y = np.array([
    [1],
    [1],
    [-1],
], dtype=float)

p = Perceptron()
p.fit(x, y)

print "perceptron"
print p.predict(np.array([[4,0], [0,0]]))
print ""


# knn
from jugg.supervised.k_nearest_neighbors import KNN

x = np.array([
    [1,0],
    [-1,0],
    [1,3],
    [2,0],
], dtype=float)

y = np.array([
    [1],
    [2],
    [3],
    [4],
], dtype=float)

knn = KNN(k=5)
knn.fit(x, y)

print "knn"
print knn.predict(np.array([[2,2], [1,-1]]))
print ""


# LR
from jugg.supervised.logistic_regression import LR

x = np.array([
    [1,1],
    [1,2],
    [-1,-1],
    [-2,-1],
], dtype=float)

y = np.array([
    [1],
    [1],
    [0],
    [0],
], dtype=float)

lr = LR()
lr.fit(x, y, iterations=100)

print "LR"
print lr.predict(np.array([[1,1], [0,0]]))
print ""


# Naive Bayes
from jugg.supervised.naive_bayes import NaiveBayes

x = np.array([
    [1,1],
    [1,2],
    [-1,-1],
    [-2,-1],
], dtype=float)

y = np.array([
    [1],
    [1],
    [2],
    [3],
], dtype=float)

nb = NaiveBayes()
nb.fit(x, y)

print "naive bayes"
print nb.predict(np.array([[1,1], [1,2]]))
print ""


# SVM
from jugg.supervised.support_vector_machine import SVM

x = np.array([
    [1,1],
    [1,2],
    [-1,-1],
    [-2,-1],
], dtype=float)

y = np.array([
    [1],
    [1],
    [-1],
    [-1],
], dtype=float)

svm = SVM()
svm.fit(x, y)

print "svm"
print svm.predict(np.array([[1,1], [-1,-1]]))
print ""


# AdaBoost
from jugg.supervised.adaboost import AdaBoost

x = np.array([
    [1,1],
    [1,2],
    [-1,-1],
    [-2,-1],
], dtype=float)

y = np.array([
    [1],
    [1],
    [-1],
    [-1],
], dtype=float)

ab = AdaBoost()
ab.fit(x, y)

print "adaboost"
print ab.predict(np.array([[1,1], [-1,-1]]))
print ""
