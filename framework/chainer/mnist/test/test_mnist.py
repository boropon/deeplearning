import unittest

import os, sys
import src.mnist as mnist

class TestMnistMethods(unittest.TestCase):

    def test_conv(self):
        batch = [[1,1]]
        mnist.conv(batch, 1)
