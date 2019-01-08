# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 16:04:28 2019

@author: 00v6834
"""

import numpy as np
import sys
sys.path.append('C:\\Users\\00v6834\\Documents\\RL_experiment')
from rl import qlearn
from rl import randAgent
from rl import neuralAgent
from rl.models import dwellScheduler
from rl.models import radar
from rl.models import rcv

dwellSch = dwellScheduler.dwellScheduler()
myRadar = radar.radar()
myRadar.RFsch = 'rand'
myRcv = rcv.rcv(dwellSch)

myRcv.setBand([0])
rewardSave =[]
for k in range(0,100):
    rf = myRadar.selectRF()
    reward = myRcv.getCount([rf])
    rewardSave.append(reward)
    myRcv.train(reward)
    myRcv.setBand([rf])
        