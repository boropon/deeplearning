import unittest
import numpy as np
from chainer import Variable

from src.affine_model import AffineModel

class TestAffineModelMethods(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_affine_model(self):
        model = AffineModel()
        model.zerograds()
        #result = model(Variable(np.zeros(784)), Variable(np.zeros(10)))
        np_array = np.array([0]*784, [0]*10)
        result = model(Variable(np_array))
        #print result
        
