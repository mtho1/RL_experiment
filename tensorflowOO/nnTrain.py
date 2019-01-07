# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 21:06:57 2018

@author: Micha
"""
import tensorflow as tf
class nnTrain:
    def __init__(self,lossFun):
        self.lossFun = lossFun
        self.lr = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.training_op = self.optimizer.minimize(self.lossFun)
        print('init nn train')
    