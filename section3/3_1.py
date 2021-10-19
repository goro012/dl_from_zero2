import numpy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))

import common


c = numpy.array([[1, 0, 0, 0, 0, 0, 0]])
W = numpy.random.rand(7, 3)
h = numpy.dot(c, W)

assert numpy.all(h == W[0,:])
