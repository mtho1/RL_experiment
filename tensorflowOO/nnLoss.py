# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 21:06:57 2018

@author: Micha
"""
import tensorflow as tf
class nnLoss:
    def __init__(self,predict):
        print('init Loss')
        self.truth = tf.placeholder(tf.float32,shape=(None))
        self.predict = predict
        self.lossFun = tf.losses.mean_squared_error(self.truth,self.predict)
        
    