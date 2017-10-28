# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 23:55:12 2017

@author: tm
"""

import tensorflow as tf
import numpy as np

sess = tf.Session()
def sample_simplenum():
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)
    print(node1, node2) #セッションで計算しないと3.0, 4.0は得られない。
    
    
    print(sess.run([node1, node2])) #3.0, 4.0が得られる
    
def sample_add():
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)
    node3 = tf.add(node1, node2)
    print("node3: ", node3)
    print("sess.run(node3): ", sess.run(node3))
         
def sample_placeholder():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b
    print(sess.run(adder_node, {a:3, b:4.5}))
    print(sess.run(adder_node, {a: [1,3], b: [2,4]}))
    
def sample_add_and_triple():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b
    add_and_triple = adder_node * 3
    print(sess.run(add_and_triple, {a: 3, b: 4.5}))
    
def sample_variable():
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    init = tf.global_variables_initializer()
    sess.run(init)    #Variableの初期化はglobal_variables_initializerをrunするまで行われない。
    print(sess.run(linear_model, {x:[1,2,3,4]}))
    
def sample_error_delta():
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)
    init = tf.global_variables_initializer()
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    sess.run(init)
    print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

def sample_error_delta2():
    W = tf.Variable([-1.], tf.float32)
    b = tf.Variable([1.], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)
    init = tf.global_variables_initializer()
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    sess.run(init)
    print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

def sample_optimizer():
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    
    init = tf.global_variables_initializer()
    optimizer = tf.train.GradientDescentOptimizer(0.01) #勾配降下法
    train = optimizer.minimize(loss)    #誤差を最小にする
    
    sess.run(init)
    for i in range(1000):
        sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
    
    print(sess.run([W, b])) #学習結果の表示

def sample_linear_train():
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    x_train = [1,2,3,4]
    y_train = [0,-1,-2,-3]
    
    init = tf.global_variables_initializer()
    optimizer = tf.train.GradientDescentOptimizer(0.01) #勾配降下法
    train = optimizer.minimize(loss)    #誤差を最小にする
    
    sess.run(init)
    for i in range(1000):
        sess.run(train, {x:x_train, y:y_train})
    
    print(sess.run([W, b, loss], {x:x_train, y:y_train})) #学習結果の表示

def sample_linear_train2(): #tf.contrib.learnライブラリを使った記述
    features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
    x = np.array([1., 2., 3., 4.])
    y = np.array([0., -1., -2., -3.])
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4, num_epochs=1000)
    estimator.fit(input_fn=input_fn, steps=1000)
    estimator.evaluate(input_fn=input_fn)

sample_simplenum()
sample_add()
sample_placeholder()
sample_add_and_triple()
sample_variable()
sample_error_delta()
sample_error_delta2()
sample_optimizer()
sample_linear_train()
sample_linear_train2()