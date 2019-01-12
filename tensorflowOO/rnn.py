# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 13:16:02 2019

@author: Micha
"""
import numpy as np
import tensorflow as tf
class rnn:
    def __init__(self,num_in,num_out,num_hidd):
        self.num_in = num_in
        self.num_out = num_out
        self.num_hidden = num_hidd
        self.inputs = tf.placeholder(tf.float32,shape=(None,num_in),name='input')
        self.lastH = None
        self._createVars()
        self.lr = 0.001
    def rnn_step(self,h_prev,x):
        initVar = tf.contrib.layers.xavier_initializer()
          
        h_prev = tf.reshape(h_prev, [1, self.num_hidden])
        x = tf.reshape(x, [1, self.num_in])
        with tf.variable_scope('rnn_block',reuse=tf.AUTO_REUSE):
            #self.W_h = tf.get_variable('W_h',shape=[self.num_hidden,self.num_hidden],initializer = initVar)
            self.W_h = tf.get_variable('W_h',shape=[1],initializer = initVar)
            self.b_h = tf.get_variable('b_h',shape=[self.num_hidden])
            self.W_x = tf.get_variable('W_x',shape=[self.num_in,self.num_hidden])
            #h =  tf.matmul(h_prev, self.W_h) + tf.matmul(x, self.W_x) + self.b_h
            h =  tf.sigmoid(self.W_h*h_prev + tf.matmul(x, self.W_x) + self.b_h)
            
            h = tf.reshape(h, [self.num_hidden], name='h')        
        return h
    def train(self,myTrain,myLoss,data_in,truth):
        # data_in: number of examples x number of features
        # truth:number of examples x number of outputs
       # print('not implemented')
        lr =self.lr
        shapeD = data_in.shape
        shapeT = truth.shape
        data_in.shape = (-1,self.num_in)
        truth.shape = (-1,self.num_out)
        if self.lastH is None:
            initVal = np.zeros(self.num_hidden)
        else:
            initVal = self.lastH
        h,_ = tf.get_default_session().run([self.states,myTrain.training_op],feed_dict={self.inputs: data_in, myLoss.truth: truth,myTrain.lr: lr,self.initial_state: initVal})  
        #myTrain.training_op.run(feed_dict={self.inputs: data_in, myLoss.truth: truth,myTrain.lr: lr,self.initial_state: initVal})
        #h = self.states.eval(feed_dict={self.inputs: data_in, self.initial_state: initVal})
        self.lastH = h[-1]
        truth.shape = shapeT
        data_in.shape = shapeD
        self.lr = max(1e-5,lr*0.99999)
    def computeLoss(self,myLoss,data_in,truth):
        shapeD = data_in.shape
        shapeT = truth.shape
        data_in.shape = (-1,self.num_in)
        truth.shape = (-1,self.num_out)
        lossVal = myLoss.lossFun.eval(feed_dict={self.inputs: data_in, myLoss.truth: truth,self.initial_state: np.zeros(self.num_hidden)})
        
        truth.shape = shapeT
        data_in.shape = shapeD
        return lossVal
    def _createVars(self):        
        with tf.variable_scope('states'):
            self.initial_state = tf.placeholder(tf.float32,shape=(self.num_hidden),name='initial_state')
            self.states = tf.scan(self.rnn_step, self.inputs,initializer=self.initial_state, name='states')
        with tf.variable_scope('output',reuse=tf.AUTO_REUSE):
            self.W_y = tf.get_variable('W_x',shape=[self.num_hidden,self.num_out])
            self.b_y = tf.get_variable('by',shape=[self.num_out])
            self.out = tf.matmul(self.states, self.W_y) + self.b_y
    def computeOutput(self,data_in):
        #out = self.outputs.eval(feed_dict={self.inputs: data_in, self.initial_state: np.zeros(self.num_hidden)})
        #h = self.states.eval(feed_dict={self.inputs: data_in, self.initial_state: np.zeros(self.num_hidden)})
        shapeD = data_in.shape
        data_in.shape = (-1,self.num_in)
        if self.lastH is None:
            (h,out)=tf.get_default_session().run([self.states,self.out],feed_dict={self.inputs: data_in, self.initial_state: np.zeros(self.num_hidden)})
        else:
            (h,out)=tf.get_default_session().run([self.states,self.out],feed_dict={self.inputs: data_in, self.initial_state: self.lastH})
        self.lastH = h[-1]
        data_in.shape = shapeD
        return (out)
    def clearHiddenState(self):
        #this is only needed for rnn
        self.lastH = None
    
    