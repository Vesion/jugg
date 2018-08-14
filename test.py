import numpy as np
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


x = np.array([
    [1,0],
    [-1,0],
], dtype=float)

y = np.array([
    [1],
    [-1],
], dtype=float)

from mlpy.supervised.perceptron import Perceptron
p = Perceptron(iterations = 100)
p.fit(x, y)
print p.predict(np.array([-1,0]))
