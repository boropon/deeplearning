"""
Cnn model
"""
import chainer.functions as F
import chainer.links as L
from chainer import Chain

class CnnModel(Chain):
    """
    Neural network(Cnn)
    """
    def __init__(self):
        super(CnnModel, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 20, 5)
            self.conv2 = L.Convolution2D(20, 50, 5)
            self.fc1 = L.Linear(800, 500)
            self.fc2 = L.Linear(500, 10)

    def __call__(self, x):
        return self.fc2(F.dropout( \
            F.relu(self.fc1( \
            F.max_pooling_2d(F.relu( \
            self.conv2(F.max_pooling_2d( \
            F.relu(self.conv1(x)), 2))), 2)))))
