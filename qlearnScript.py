# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 20:58:22 2018

@author: Micha
"""
#load_ext autoreload
#autoreload 2
import numpy as np
#from  importlib import reload 
from rl.models import tictacToe
from rl import qlearn
from rl import randAgent
from rl import neuralAgent
import copy
#reload(tictacToe)
#reload(qlearn)
#agent = qlearn.qlearn(9,3**9,0.2)
agent = neuralAgent.neuralAgent(9,9,0.2)
alt_agent = randAgent.randAgent(9,3**9)
game = tictacToe.tictacToe(alt_agent)

count = 0
wins = 0
loss = 0
tie = 0
for episode in range(1,100001):
    game.resetGame()
    agent.validateAllActions()
    while game.gameOver == False:
        if agent.agent_type == 'neural':
            currentState = game.board
        else:
            currentState = game.encodeBoardState()
            
            
        validMove = False
        while validMove == False:
            action = agent.chooseActionEpsGreedy(currentState)
            (p1,p2) = game.decodeAction(action)
            validMove = game.moveAgent(p1,p2)
            if validMove == False:
                agent.invalidateLastAction()
                
        if agent.agent_type == 'neural':
            newState = game.board
        else:
            newState = game.encodeBoardState()
        if game.terminal:
            agent.update(game.lastReward,-1)
        else:
            agent.update(game.lastReward,newState)
    if game.status == 2:
        wins += 1
    elif game.status ==1:
        loss += 1
    elif game.status == 0:
        tie += 1
    else:
        raise Exception('invlaid status')
    if np.mod(episode,100)==0:
        print((wins,loss,tie))
        print(wins/(wins+loss+tie))
        print(loss/(wins+loss+tie))
        print(episode)
        wins = 0
        loss = 0
        tie = 0
    if episode >= np.inf and np.mod(episode,100)==0:
        alt_agent = copy.deepcopy(agent)
        alt_agent.eps = 0.1;
        game = tictacToe.tictacToe(alt_agent)
    agent.eps = agent.eps * 0.9999 
            
            
                
                
        
            
    



    

