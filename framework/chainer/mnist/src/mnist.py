"""
MNIST training program using chainer.
"""
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers, Chain

from affine_model import AffineModel
from cnn_model import CnnModel
from network import Network

# load MNIST dataset
TRAIN, TEST = chainer.datasets.get_mnist(ndim=3)

MODEL = L.Classifier(CnnModel())
OPTIMIZER = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
OPTIMIZER.setup(MODEL)

NETWORK = Network(MODEL, OPTIMIZER, TRAIN, TEST)
for i in range(20):
    NETWORK.train()
    print i, NETWORK.test()
