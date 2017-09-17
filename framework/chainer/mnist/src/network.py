"""
Neural network
"""
import numpy as np

import chainer.iterators as I
from chainer import Variable

class Network(object):
    """
    Neural network
    """
    def __init__(self, model, optimizer, train_data, test_data):
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data
        self.test_data = test_data

    @staticmethod
    def conv(batch, batchsize):
        """
        Separate batch into input and teature
        """
        x = []
        t = []
        for i in range(batchsize):
            x.append(batch[i][0])
            t.append(batch[i][1])
        return Variable(np.array(x)), Variable(np.array(t))

    def test(self, batchsize=1000):
        """
        Check loss
        """
        test_batch = I.SerialIterator(self.test_data, batchsize, repeat=False).next()
        x, t = self.conv(test_batch, batchsize)
        loss = self.model(x, t)
        return loss.data

    def train(self, batchsize=1000):
        """
        Train neural network
        """
        for train_batch in I.SerialIterator(self.train_data, batchsize, repeat=False):
            x, t = self.conv(train_batch, batchsize)

            self.model.zerograds()
            loss = self.model(x, t)
            loss.backward()
            self.optimizer.update()
