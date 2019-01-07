# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 07:50:12 2019

@author: Micha
"""

import numpy as np
class randAgent:
    def __init__(self,num_actions,num_states):
        self.num_actions = num_actions
        self.num_states = num_states
        self.validAction = np.ones((num_states,num_actions))
        self.lastState = []
        self.lastAction = []
    def chooseActionEpsGreedy(self,currentState):
        action = self.chooseActionGreedy(currentState)
        return action
    def chooseActionGreedy(self,currentState):
        validActionsTemp = np.where(self.validAction[currentState,:])
        L =  len(validActionsTemp[0])
        temp = np.random.randint(0,L)
        action = validActionsTemp[0][temp]
        self.lastState = currentState
        self.lastAction = action
        return action    
    def invalidateLastAction(self):
        self.validAction[self.lastState,self.lastAction] = 0
    def validateAllActions(self):
        #print('not implemented')
        self.validAction = np.ones((num_states,num_actions))
    def update(self,reward,nextState):
        #use -1 for nextState if we arrive at terminal state
        print('not used here')
    
        
    
            
    
    