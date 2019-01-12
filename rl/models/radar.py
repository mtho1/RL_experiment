# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:44:38 2019

@author: 00v6834
"""
import numpy as np
class radar:
    def __init__(self):
        print('init Radar2')
        self.rf_st = 1
        self.rf_end = 1.1
        self.rf_spac = 0.01
        self.numRF = np.round( (self.rf_end-self.rf_st)/self.rf_spac ) + 1
        self.RFsch = 'rand'  #a string "rand", "order", "const"
        self.lastChannel = None
    def selectRF(self,num=1):
        if self.RFsch == 'rand':
            RFind = np.random.randint(0,self.numRF)
        elif self.RFsch == 'order':
            if self.lastChannel == None:
                RFind = np.random.randint(0,self.numRF)
            else:
                RFind = np.mod(self.lastChannel+1,self.numRF)
        elif self.RFsch == 'const':
            if self.lastChannel == None:
                RFind = np.random.randint(0,self.numRF)
            else:
                RFind = self.lastChannel
                
        else:
            raise(Exception('invalid RFsch'))
        self.lastChannel = RFind
        return self.mapChannel(self.lastChannel)
    def mapChannel(self,channel):
        if (channel > self.numRF-1 ) or channel < 0:
            raise(Exception('invalid channel'))
        RF = self.rf_st + self.rf_spac*(channel-1)
        return RF
    
        
        
            
                
        
    
