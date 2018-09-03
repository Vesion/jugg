import numpy as np

# ## utils
#
# from mlpy.utils.activation_functions import *
# from mlpy.utils.loss_functions import *
# from mlpy.utils.metrics import *
# from mlpy.utils.optimizers import *

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
from mlpy.supervised.perceptron import Perceptron

x = np.array([
    [1,0],
    [-1,0],
], dtype=float)

y = np.array([
    [1],
    [-1],
], dtype=float)

p = Perceptron()
p.fit(x, y)

print "perceptron"
print p.predict(np.array([[1,0], [-1,0]]))
print ""


# knn
from mlpy.supervised.k_nearest_neighbors import KNN

x = np.array([
    [1,0],
    [-1,0],
], dtype=float)

y = np.array([
    [4],
    [1],
], dtype=float)

knn = KNN(k=5)
knn.fit(x, y)

print "knn"
print knn.predict(np.array([[2,0], [1,0]]))
print ""


# LR
from mlpy.supervised.logistic_regression import LR

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
from mlpy.supervised.naive_bayes import NaiveBayes

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
    [2],
], dtype=float)

nb = NaiveBayes()
nb.fit(x, y)

print "naive bayes"
print nb.predict(np.array([[1,1], [1,2]]))
print ""


# SVM
from mlpy.supervised.support_vector_machine import SVM

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

