# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:38:12 2017

@author: tm
"""

import tensorflow as tf

"""
Super Resolution using Convolutional Neural Network
"""
class SRCNN(object):
    """
    オブジェクト生成時に計算グラフを構築する。計算処理は行わない。
    """
    def __init__(self,
                 image_size=33,
                 label_size=21,
                 depth=1):
        self.image_size = image_size
        self.label_size = label_size
        self.depth = depth
        
        self.build_model()
        
    """
    計算グラフの構築
    """
    def build_model(self):
        
        # 訓練画像と期待画像
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.depth], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.depth], name='labels')
        
        # Weight
        self.w1 = tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1')
        self.w2 = tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2')
        self.w3 = tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
        
        # Bias
        self.b1 = tf.Variable(tf.zeros([64]), name='b1')
        self.b2 = tf.Variable(tf.zeros([32]), name='b2')
        self.b3 = tf.Variable(tf.zeros([1]), name='b3')
        
        # 畳み込み演算
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.w1, strides=[1,1,1,1], padding='VALID') + self.b1)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.w2, strides=[1,1,1,1], padding='VALID') + self.b2)
        conv3 = tf.nn.conv2d(conv2, self.w3, strides=[1,1,1,1], padding='VALID') + self.b3
        
        # 損失計算(MSE)
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        
        # 訓練結果の保存
        self.saver = tf.train.Saver()
        
    def train(self):
        
    def test(self):
        
    def forward(self):
        
    def save(self):
        
    def load(self):
        