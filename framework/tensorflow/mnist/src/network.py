# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:25:32 2017

@author: tm
"""

"""
ニューラルネットワーク
"""
import tensorflow as tf

class Network(object):
    """
    Neural network
    """
    def __init__(self, mnist, model="conv", optimizer="momentum"):
        self.model = model
        self.optimizer = optimizer
        self.mnist = mnist
        self.sess = tf.InteractiveSession()

    def __weight_variable(self, shape):
        """
        標準正規分布に従うランダム値で初期化されるtf.Variableを取得
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def __bias_variable(self, shape):
        """
        0.1で初期化されるtf.Variableを取得
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def __conv2d(self, x, W, stride):
        """
        畳み込み演算
            x : 入力データ
            W : フィルターサイズ(幅, 高さ, 入力チャンネル数, 出力チャンネル数)
            stride : ストライド幅、高さ
        """
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

    def __max_pool(self, x, size, stride):
        """
        プーリング
        """
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='VALID')

    def set_conv_graph(self):
        self.x = tf.placeholder("float", shape=[None, 784])
        self.y_ = tf.placeholder("float", shape=[None, 10])
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        
        self.W_conv1 = self.__weight_variable([5, 5, 1, 20])
        self.W_conv2 = self.__weight_variable([5, 5, 20, 50])
        self.W_fc1 = self.__weight_variable([800, 500])
        self.W_fc2 = self.__weight_variable([500, 10])
        self.b_conv1 = self.__bias_variable([20])
        self.b_conv2 = self.__bias_variable([50])
        self.b_fc1 = self.__weight_variable([500])
        self.b_fc2 = self.__weight_variable([10])
        
        #1層目
        self.h1 = self.__max_pool(tf.nn.relu(self.__conv2d(self.x_image, self.W_conv1, 1) + self.b_conv1), 2, 2)
        #2層目
        self.h2 = self.__max_pool(tf.nn.relu(self.__conv2d(self.h1, self.W_conv2, 1) + self.b_conv2), 2, 2)
        #3層目
        self.h3 = tf.nn.relu(tf.matmul(tf.reshape(self.h2, [-1, 800]), self.W_fc1) + self.b_fc1)
        #4層目
        self.y = tf.matmul(self.h3, self.W_fc2) + self.b_fc2
        
        #学習
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
        self.train_step = tf.train.MomentumOptimizer(0.01, 0.9).minimize(self.cross_entropy)
        
        #推測
        self.prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.prediction, tf.float32))

    def train(self):
        """
        学習
        """
        tf.global_variables_initializer().run()
        for _ in range(50):
            batch = self.mnist.train.next_batch(1000)
            self.sess.run(self.train_step, feed_dict={self.x: batch[0], self.y_: batch[1]})
        
    def test(self):
        """
        テスト
        """
        print('test accuracy %g' % self.sess.run(self.accuracy, feed_dict={self.x: self.mnist.test.images, self.y_: self.mnist.test.labels}))