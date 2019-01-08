# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 16:27:56 2019

@author: 00v6834
"""
import numpy as np
class rcv:
    def __init__(self,dwell_agent):
        self.dwell_agent = dwell_agent
        self.currentRF = None
        self.currentBW = None
    def getCount(self,freqs):
        c1 = freqs < self.currentRF + self.currentBW/2.0
        c2 = freqs > self.currentRF - self.currentBW/2.0
        count = len(np.where(np.logical_and(c1,c2))[0])
        return count
    def setBand(self,inputVals):
        (rf,bw) = self.dwell_agent.setDwell(inputVals)
        self.currentRF = rf
        self.currentBW = bw
    def train(self,reward):
        self.dwell_agent.train(reward)
        
        
        
        
        
        