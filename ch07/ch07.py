# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 01:44:25 2017

@author: tm
"""

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from my_convnet import SimpleConvNet
from common.trainer import Trainer

from PIL import Image

# 処理に時間のかかる場合はデータを削減 
#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

#################################
# 学習処理                        
#################################
#(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
#max_epochs = 20
#trainer = Trainer(network, x_train, t_train, x_test, t_test,
#                  epochs=max_epochs, mini_batch_size=100,
#                  optimizer='Adam', optimizer_param={'lr': 0.001},
#                  evaluate_sample_num_per_epoch=1000)
#trainer.train()
#network.save_params("my_ch07_params.pkl")
#print("Saved Network Parameters!")
#markers = {'train': 'o', 'test': 's'}
#x = np.arange(max_epochs)
#plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
#plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
#plt.xlabel("epochs")
#plt.ylabel("accuracy")
#plt.ylim(0, 1.0)
#plt.legend(loc='lower right')
#plt.show()

#################################
# 認識処理                        
#################################
# 学習済みパラメータのロード
network.load_params("my_ch07_params.pkl")

for i in range(0,10):
    filename = str(i) + '_np.png'
    np_img = np.asarray(Image.open(filename)).reshape(1,1,28,28)
    
    predict = network.predict(np_img)
    print(predict)
    answer = np.argmax(predict)
    print(filename + " is " + str(answer) + ".")