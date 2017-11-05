# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:11:55 2017

@author: tm
"""

"""
MNIST training program using TensorFlow
"""
from tensorflow.examples.tutorials.mnist import input_data
from network import Network

# load MNIST dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

network = Network(mnist=mnist)
network.set_conv_graph()
network.train()
network.test()