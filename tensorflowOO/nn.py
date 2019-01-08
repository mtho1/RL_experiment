# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 20:26:38 2018

@author: Micha
"""
import tensorflow as tf
class slnn:
    #single layer nn
    def __init__(self,num_in,num_out,num_hidd):
        self.num_in = num_in
        self.num_out = num_out
        self.num_hidden = num_hidd
        self.scope = 'myNN'
        #inititializer = tf.initializers.random_normal(0,1e-10)
        self.input = tf.placeholder(tf.float32,shape=(None,num_in),name='input') #None is auto determined
        self.hid = tf.contrib.layers.fully_connected(self.input, self.num_hidden, scope='hidL',activation_fn=tf.nn.sigmoid)
        self.hid2 = tf.contrib.layers.fully_connected(self.hid, self.num_hidden, scope='hidL2',activation_fn=tf.nn.sigmoid)
        self.out = tf.contrib.layers.fully_connected(self.hid2, self.num_out, scope='outL',activation_fn=None)
        self.lr = 0.001
    def train(self,myTrain,myLoss,data_in,truth):
        # data_in: number of examples x number of features
        # truth:number of examples x number of outputs
       # print('not implemented')
        lr =self.lr
        shapeD = data_in.shape
        shapeT = truth.shape
        data_in.shape = (-1,self.num_in)
        truth.shape = (-1,self.num_out)
        myTrain.training_op.run(feed_dict={self.input: data_in, myLoss.truth: truth,myTrain.lr: lr})
        truth.shape = shapeT
        data_in.shape = shapeD
        self.lr = max(1e-5,lr*0.99999)
    def computeLoss(self,myLoss,data_in,truth):
        lossVal = myLoss.lossFun.eval(feed_dict={self.input: data_in, myLoss.truth: truth})
        return lossVal
    
    def computeOutput(self,data_in):
        out = self.out.eval(feed_dict={self.input: data_in})
        return out
        
    