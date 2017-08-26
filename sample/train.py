# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 00:07:35 2017

@author: tm
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import argparse

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from my_mnist import load_mnist
from my_trainer import Trainer
from my_deep_convnet import DeepConvNet
from my_convnet import SimpleConvNet
from my_two_layer_net import TwoLayerNet

############################################################
# オプション解析
############################################################
def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='affine', choices=['deepconv', 'simpleconv', 'affine'])
    parser.add_argument('--output', type=str, default='learning_result.pkl')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Momentum', 'Nesterov', 'AdaGrad', 'RMSprpo', 'Adam'])
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
# 学習処理
############################################################
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

args = parse_args()
network = get_network(args.network)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=args.epochs, mini_batch_size=100,
                  optimizer=args.optimizer, optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)

trainer.train()
network.save_params(args.output)

print("Saved Network Parameters!")
markers = {'train': 'o', 'test': 's'}
x = np.arange(args.epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()