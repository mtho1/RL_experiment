# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 07:50:12 2019

@author: Micha
"""

import numpy as np
class qlearn:
    def __init__(self,num_actions,num_states,eps):
        self.num_actions = num_actions
        self.num_states = num_states
        self.Q = np.random.rand(num_states,num_actions)*0.001
        self.eps = eps #for epsilon greedy
        self.lr = 0.1
        self.lastAction = []
        self.lastState = []
        self.lastReward = []
        self.discount = 0.9
        self.agent_type = 'qlearn'
    def chooseActionEpsGreedy(self,currentState):
        self.lastState = currentState
        if np.random.rand() > self.eps:
            action = np.argmax(self.Q[currentState,:])
        else:
            action = np.random.randint(0,self.num_actions)
        self.lastAction = action
        return action
    def chooseActionGreedy(self,currentState):
        action = np.argmax(self.Q[currentState,:])
        self.lastState = currentState
        self.lastAction = action
        return action    
    def invalidateLastAction(self):
        self.Q[self.lastState,self.lastAction] = -np.Inf
    def update(self,reward,nextState):
        #use -1 for nextState if we arrive at terminal state
        self.lastReward = reward
        target = self.lastReward + self.discount*np.max(self.Q[nextState,:]) 
        self.Q[self.lastState,self.lastAction] = (1-self.lr)*self.Q[self.lastState,self.lastAction] + self.lr*target
    
            
    
    