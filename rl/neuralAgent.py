# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 07:50:12 2019

@author: Micha
"""
from tensorflowOO import *
import numpy as np
import tensorflow as tf
from rl import ExperienceBuffer
class neuralAgent:
    def __init__(self,num_actions,num_states,eps):
        defaultSess = tf.get_default_session()
        if defaultSess is not None:
            defaultSess.close() # close previous session
            
        
        tf.reset_default_graph()
        self.num_actions = num_actions
        self.num_states = num_states
        self.validAction = np.ones((1,num_actions))
        self.lastState = []
        self.lastAction = []
        self.lr = 0.2
        self.num_hidden = 100
        self.nn = nn.slnn(self.num_states,self.num_actions,self.num_hidden)
        self.myLoss = nnLoss.nnLoss(self.nn.out)
        self.myOptim = nnTrain.nnTrain(self.myLoss.lossFun)
        self.tf_sess = tf.InteractiveSession()
        self.discount = 0.9
        self.eps = eps
        self.lastQ = []
        tf.global_variables_initializer().run()
        self.agent_type = 'neural'
        self.experienceBuffer = ExperienceBuffer.ExperienceBuffer(200) 
        print('neuralAgent init3')
    def evalQ(self,state):
        #state should be 1 x self.num_states
        stateCopy = np.copy(state)
        stateCopy.shape = (1,self.num_states)
        Q = self.nn.computeOutput(stateCopy)
        return Q
    def chooseActionEpsGreedy(self,currentState):
        if np.any(currentState != self.lastState):
            self.validateAllActions()                    
        if np.random.rand() > self.eps:
            action = self.chooseActionGreedy(currentState)
            self.lastQ = self.evalQ(currentState)
        else:
            validActionsTemp = np.where(self.validAction[0,:])
            L =  len(validActionsTemp[0])
            temp = np.random.randint(0,L)
            action = validActionsTemp[0][temp]        
        #print('not done')
        self.lastAction = action
        self.lastState = np.copy(currentState)
        return action
    def chooseActionGreedy(self,currentState):
        if np.any(currentState != self.lastState):
            self.validateAllActions()
        self.lastQ = self.evalQ(currentState)
        action = []
        validActionsTemp = np.where(self.validAction[0,:])[0]
        Q = np.copy(self.lastQ)
        Q = Q[0,validActionsTemp]
        actionInd = np.argmax(Q)
        action = validActionsTemp[actionInd]
        self.lastAction = action
        self.lastState = np.copy(currentState)
        return action    
    def invalidateLastAction(self):
       # print('not implemented')
        self.validAction[0,self.lastAction] = 0
    def validateAllActions(self):
        #print('not implemented')
        self.validAction[0,:] = 1
    def update(self,reward,nextState):
        #print('not done')
        oldVal = self.evalQ(self.lastState)
        target = np.copy(oldVal)
        terminalFlag = False
        if np.size(nextState) == 1 : # 
            if np.all(nextState == -1):  #then terminal
                target[0,self.lastAction] = 0
                terminalFlag = True
            else:
                temp = self.discount*self.evalQ(nextState)
                target[0,self.lastAction] = temp[0,self.lastAction]
        else:
            temp = self.discount*self.evalQ(nextState)
            target[0,self.lastAction] = temp[0,self.lastAction]
        
            
            
        target[0,self.lastAction] += reward
        target = (1-self.lr)*oldVal + self.lr*target
        #print('still need train')
        for  k in range(0,5):
            self.nn.train(self.myOptim,self.myLoss,self.lastState,target)
        #if not terminalFlag:
        self.experienceBuffer.addExperience(self.lastState,nextState,reward,self.lastAction)
        self.replayExperience(20)
    def replayExperience(self,numReplay):
        lastState =[]
        target =[]
        for it in range(0,numReplay):
            lastStateTemp,newState,reward,lastAction = self.experienceBuffer.replay()
            terminalFlag = False
            if np.size(newState) == 1 : # 
                if np.all(newState == -1):  #then terminal
                    terminalFlag = True
            
            lastState.append(lastStateTemp)
            target.append(self.evalQ(lastState[it])[0])
            #target = np.copy(oldVal)
            if not terminalFlag: 
                temp = self.discount*self.evalQ(newState) 
                temp = temp[0,lastAction]+ reward
            else:
                temp = reward 
                    
            temp = (1-self.lr)*target[it][lastAction] + self.lr*temp
            #target[0,lastAction] = temp[0,lastAction]
            #target[0,lastAction] += reward
            target[it][lastAction] = temp
        self.nn.train(self.myOptim,self.myLoss,np.array(lastState),np.array(target))
        
    def __del__(self):
        tf.InteractiveSession.close(self.tf_sess)
        
            
    
    