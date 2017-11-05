# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 12:43:19 2017

@author: tm
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

def weight_variable(shape):
    '''
    重みの初期化
        tf.truncated_normal()は標準正規分布に従うランダム値を設定する。
            shape : 1次元のテンソルかリスト。出力テンソルの形式。
            mean : 平均値。初期値は0.0。
            stddev : 標準偏差。初期値は1.0。
            dtype : 出力のデータ型。初期値はtf.float32。
            seed : ランダムのシード値。初期値はNone。
            name : 処理の名前。初期値はNone。
    '''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    '''
    バイアスの初期化
        tf.constant()は定数を設定する。
            value : 定数値。
            dtype :　出力のデータ型。初期値はNone。
            shape : 出力テンソルの形式。
            name : テンソルの名前
            verify_shape : 出力テンソルの形式をVerificationする。
    '''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    '''
    畳み込み演算
        tf.nn.conv2d()は畳み込み演算を行う。
            input : 入力データ。4次元のテンソル(batch, in_height, in_width, in_channels)。
            filter : フィルター。4次元のテンソル(filter_height, filter_width, in_channels, channel_multipier)。
            strides : ストライド。[1, stride, stride, 1]
            padding : パディングアルゴリズム。SAME or VALID。
            name : 処理の名前。初期値はNone。
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    '''
    プーリング
        tf.nn.max_pool()はプーリングを行う。
            value : 入力データ。4次元テンソル。
            ksize : ウィンドウサイズ。[1, size, size, 1]
            strides : ウィンドウのストライド。[1, stride, stride, 1]
            padding : パディングアルゴリズム。SAME or VALID。
            name : 処理の名前。初期値はNone。
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

'''
畳み込み1層
    フィルタサイズ
        幅 : 5
        高さ : 5
        入力チャンネル数 : 1
        出力チャンネル数 : 32
    ストライド
        横 : 1
        縦 : 1
    イメージサイズ
        28x28 -> 28x28 -> 14x14
'''
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

'''
畳み込み2層
    フィルタサイズ
        幅 : 5
        高さ : 5
        入力チャンネル数 : 32
        出力チャンネル数 : 64
    ストライド
        横 : 1
        縦 : 1
    イメージサイズ
        14x14 -> 14x14 -> 7x7
'''
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''
全結合1層

    幅7 * 高さ7 * 64チャンネル -> 1024
'''
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

'''
ドロップアウト


'''
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

'''
全結合2層

    1024 -> 10
'''
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

'''
学習

    交差エントロピーの演算。
    Adamでの学習。
    学習結果の評価。
'''
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.image, y_: mnist.test.labels, keep_prob:1.0}))