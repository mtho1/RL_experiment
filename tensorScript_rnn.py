# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 20:58:22 2018

@author: Micha
"""
import numpy as np
from  importlib import reload 
import tensorflow as tf
tf.reset_default_graph()
from tensorflowOO import rnn
from tensorflowOO import nnLoss
from tensorflowOO import nnTrain

#if tf.get_default_session() is None:
defaultSess = tf.get_default_session()
if defaultSess is not None:
    defaultSess.close() # close previous session
sess2 = tf.InteractiveSession()



myNet=rnn.rnn(3,2,10)
myLoss = nnLoss.nnLoss(myNet.out)
myOptim = nnTrain.nnTrain(myLoss.lossFun)



init = tf.global_variables_initializer()
init.run()

x = np.random.rand(100,3)
x[:,0] = np.linspace(-3,3,100)
x[:,1] = np.linspace(-3,3,100)**2
x[:,2] = x[:,0]+0.5*x[:,1]
y = np.random.rand(100,2)
y[:,0] =np.sin(x[:,0]) - np.cos(x[:,1])
y[:,1] =x[:,0]

#y=np.sin(x[:,0])+x[:,1]
for it in range(0,1000):
    myNet.train(myOptim,myLoss,x,y)
    print(myNet.computeLoss(myLoss,x,y))
(o,h) = myNet.computeOutput(x)

W_h = myNet.W_h.eval()
W_x = myNet.W_x.eval()
b_h = myNet.b_h.eval()


#k =0 
#while k<1000:
#    myNet.train(myOptim,myLoss,data,truth)
#    print(myNet.computeLoss(myLoss,data,truth))
#    k += 1
#sess2.close()
#myNet.computeLoss(myLoss,data,truth)
    


