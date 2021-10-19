import numpy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))

import common
import common.layers


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        W_in = 0.01 * numpy.random.rand(V, H).astype("f")
        W_out = 0.01 * numpy.random.rand(H, V).astype("f")

        self.in_layer0 = common.layers.MatMul(W_in)
        self.in_layer1 = common.layers.MatMul(W_in)
        self.out_layer = common.layers.MatMul(W_out)
        self.loss_layer = common.layers.SoftmaxWithLoss()

        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:,0])
        h1 = self.in_layer1.forward(contexts[:,1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da += 0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)
        return None