# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 08:00:46 2019

@author: Micha
"""
import numpy as np
class ExperienceBuffer:
    def __init__(self,MaxBufferSize):
        self.MaxBufferSize = MaxBufferSize 
        self.state0 = []  #list of previous state
        self.state1 = [] #list of next state
        self.reward = [] #list of reward
        self.action = [] #list of actions
        self.bufferSize = 0
    def addExperience(self,s0,s1,reward,action):
        if self.bufferSize >= self.MaxBufferSize:
            indPop = np.random.randint(0,self.bufferSize)
            self.state0[indPop]= s0
            self.state1[indPop]= s1
            self.reward[indPop]= reward
            self.action[indPop]= action
        else:
            self.bufferSize += 1
            self.state0.append(s0)
            self.state1.append(s1)
            self.reward.append(reward)
            self.action.append(action)
    def replay(self):
        #returns(s0,s1,reward,act randomly selected
        ind = np.random.randint(0,self.bufferSize)
        s0 = self.state0[ind]
        s1 = self.state1[ind]
        reward = self.reward[ind]
        act = self.action[ind]
        return (s0,s1,reward,act)
    
    
    