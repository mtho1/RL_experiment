# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 14:08:30 2019

@author: 00v6834
"""
import numpy as np
from rl import neuralAgent
class dwellScheduler:
    def __init__(self):
        self.minRF = 0
        self.maxRF = 3
        self.spacRF = 0.1
        self.bandwidths = np.array( [0.02 ,0.05, 0.1, 0.2, 0.5] )
        ####
        self.numRF = np.int(np.round( (self.maxRF-self.minRF)/self.spacRF + 1))
        self.numBW = len(self.bandwidths)
        self.num_actions = self.numRF*self.numBW
        ### 
        self.agent = neuralAgent.neuralAgent(self.num_actions,self.numRF,0.2)
        
    def mapAction(self,action):
        bwInd = np.floor(action/self.numRF)
        rfInd = action - bwInd*self.numRF
        bw = self.bandwidths[int(bwInd)]
        rf = self.minRF + self.spacRF*(rfInd-1)
        return (rf,bw)
    
    def mapFreq(self,freqs):
        #maps received freq to input for neural net
        freqSort = np.sort(freqs)
        freqSort = np.minimum(self.num_actions-1,np.maximum(0,np.round( (freqSort + 0.5*self.spacRF)/self.spacRF)))
        mapped = np.zeros((1,self.numRF))
        for f in freqSort:
            mapped[0,int(f)] += 1
        return mapped
    
    def setDwell(self,inputVal):
        inputValTrans = self.mapFreq(inputVal)
        action = self.agent.chooseActionEpsGreedy(inputValTrans)
        (rf,bw) = self.mapAction(action)
        return (rf,bw)
    
    def train(self,reward):
        self.agent.update(reward,-1)   # here we treat every state as terminal
        
        
    
        
        
        
    