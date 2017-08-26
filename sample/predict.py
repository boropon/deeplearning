# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 22:09:09 2017

@author: tm
"""

import sys, os
import numpy as np
import argparse

from my_deep_convnet import DeepConvNet
from my_convnet import SimpleConvNet
from my_two_layer_net import TwoLayerNet
from PIL import Image

############################################################
# オプション解析
############################################################
def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='deepconv', choices=['deepconv', 'simpleconv', 'affine'])
    parser.add_argument('--input', type=str, default='learning_result_deep.pkl')
    args = parser.parse_args()
    return args

############################################################
# ネットワークの取得
############################################################
def get_network(network):
    if network == 'deepconv':
        print('deepconv')
        network = DeepConvNet(input_dim=(1,28,28), 
                        conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                        conv_param_2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                        conv_param_3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                        conv_param_4 = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                        conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                        conv_param_6 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                        hidden_size=100, output_size=10)
    elif network == 'simpleconv':
        print('simpleconv')
        network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
    elif network == 'affine':
        print('affine')
        network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, weight_init_std = 0.01)
    else:
        print('other')
        
    return network

############################################################
# 推論処理
############################################################
args = parse_args()
network = get_network(args.network)

network.load_params(args.input)

for i in range(0,10):
    filename = str(i) + '_np.png'
    np_img = np.asarray(Image.open("../data/" + filename)).reshape(1,1,28,28) / 255.0
    
    predict = network.predict(np_img)
    print(predict)
    answer = np.argmax(predict)
    print(filename + " is " + str(answer) + ".")