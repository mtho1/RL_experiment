# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 20:58:22 2018

@author: Micha
"""
import numpy as np
from  importlib import reload 
import tensorflow as tf
tf.reset_default_graph()
from tensorflowOO import *
sess2 = tf.InteractiveSession()
reload(nn)
reload(nnTrain)
reload(nnLoss)



myNet = nn.slnn(2,1,1000)
myLoss = nnLoss.nnLoss(myNet.out)
myOptim = nnTrain.nnTrain(myLoss.lossFun)


data = np.random.rand(100,2)
data = np.reshape(data,(-1,2))   # number of examples x number of features
data2 = np.sum(data,1)
data2 =np.reshape(data2,(-1,1))
truth1 = np.sin(data2) + np.random.randn(data2.shape[0],data2.shape[1])*.00 + data2
truth2 = np.cos(data) + np.random.randn(data.shape[0],data.shape[1])*.00 - data
truth = truth1 + np.prod(truth2,1,keepdims=True) + np.random.randn(truth1.shape[0],truth1.shape[1])*.0


init = tf.global_variables_initializer()
init.run()
k =0 
while k<1000:
    myNet.train(myOptim,myLoss,data,truth)
    print(myNet.computeLoss(myLoss,data,truth))
    k += 1
#sess2.close()
myNet.computeLoss(myLoss,data,truth)
    


