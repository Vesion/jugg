import numpy as np

from jugg.deeplearning.layers import *
from jugg.utils.optimizers import SGD

dense = Dense(4, (8,))
dense.initialize(SGD)
print dense.name
print dense.parameter_size()
print dense.forward(np.ones((2, 8)))
print dense.backward(np.ones((2, 4)))
print dense.output_shape()
print ""

resh = Reshape((2,3))
print resh.name
print resh.parameter_size()
print resh.forward(np.ones((2, 6)))
print resh.backward(np.ones((2, 2, 3)))
print resh.output_shape()
