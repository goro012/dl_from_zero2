import numpy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))

import common
import common.layers


c0 = numpy.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = numpy.array([[0, 0, 1, 0, 0, 0, 0]])

W_in = numpy.random.rand(7, 3)
W_out = numpy.random.rand(3, 7)

in_layer0 = common.layers.MatMul(W_in)
in_layer1 = common.layers.MatMul(W_in)
out_layer = common.layers.MatMul(W_out)

h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)
print(h.shape)
s = out_layer.forward(h)

print(s)
