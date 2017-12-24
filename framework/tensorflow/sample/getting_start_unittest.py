# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 22:22:29 2017

@author: tm
"""

import unittest
import mock
import tensorflow as tf

import getting_start_with_tensorflow as target

class TestTensorCalculation(unittest.TestCase):
     
    @mock.patch('tensorflow.Session.run')
    @mock.patch('tensorflow.constant')
    def test_simplenum(self, constant, run):
        target.sample_simplenum()
        calls = [mock.call(3.0, tf.float32), mock.call(4.0)]
        constant.asssert_has_calls(calls)
        run.assert_called()
        
    @mock.patch('tensorflow.Session.run')
    @mock.patch('tensorflow.add')
    @mock.patch('tensorflow.constant')
    def test_add(self, constant, add, run):
        target.sample_add()
        calls = [mock.call(3.0, tf.float32), mock.call(4.0)]
        constant.asssert_has_calls(calls)
        add.assert_called()
        run.assert_called()
