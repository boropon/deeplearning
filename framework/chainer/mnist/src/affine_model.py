"""
Affine model
"""
import chainer.functions as F
import chainer.links as L
from chainer import Chain

class AffineModel(Chain):
    """
    Neural network(affine)
    784 - 100 - 100 - 10
    """
    def __init__(self):
        super(AffineModel, self).__init__()
        with self.init_scope():
            self.linear1 = L.Linear(784, 100)
            self.linear2 = L.Linear(100, 100)
            self.linear3 = L.Linear(100, 10)

    def __call__(self, x):
        return self.linear3(F.relu(self.linear2(F.relu(self.linear1(x)))))
