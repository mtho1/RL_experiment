# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 18:44:29 2018

@author: Micha
"""
import numpy as np
class tictacToe:
    def __init__(self,alt_agent):
        self.board=np.zeros((3,3)) #-1 Alt Agent, 0 empty, 1 Agent
        self.lastReward = 0
        self.gameOver = False
        self.terminal = False
        self.status = []
        self.alt_agent = alt_agent
    def resetGame(self):
        self.board=np.zeros((3,3)) #-1 Alt Agent, 0 empty, 1 Agent
        self.lastReward = 0
        self.gameOver = False
        self.terminal = False
        if np.random.rand() > 0.5:
            #then alt agent moves first
             currentState = self.encodeBoardState()
             act = self.alt_agent.chooseActionEpsGreedy(currentState)
             (p1,p2) = self.decodeAction(act)
             self.moveAltAgent(p1,p2)
        self.status = self.checkBoard()
    def checkBoard(self):
        #retuns 1 if alt-agent wins,2 if Agent wins, 0 draw,-1 no winner yet
        flagFull=False
        if np.all(self.board!=0):
            flagFull=True
        #check horizontals
        for k in range(0,3):
            if np.all(self.board[k,:]!=0):
                if np.all(self.board[k,:]==-1):
                    return 1
                elif np.all(self.board[k,:]==1):
                    return 2
        #check verticals
        for k in range(0,3):
            if np.all(self.board[:,k]!=0):
                if np.all(self.board[:,k]==-1):
                    return 1
                elif np.all(self.board[:,k]==1):
                    return 2
        #check diagonal
        diag1=np.diag(self.board)
        diag2=np.diag(np.flipud(self.board))
        if np.all(diag1!=0):
            if np.all(diag1==-1):
                return 1
            elif np.all(diag1==1):
                return 2
        if np.all(diag2!=0):
            if np.all(diag2==-1):
                return 1
            if np.all(diag2==1):
                return 2
            #ok no winner
        if flagFull:
            return  0   #draw
        else:
            return -1   # game continues
    def genRand(self):
        valid=False
        while not valid:
            pos1=np.random.randint(0,high=3,size=1)
            pos2=np.random.randint(0,high=3,size=1)
            if self.board[pos1,pos2]==0:
                valid=True
        return((pos1,pos2))
    def moveAgent(self,pos1,pos2):
        mark = 1
        valid=False
        if pos1>=0 and pos1 <3 and pos2>=0 and pos2 <3 and self.board[pos1,pos2]==0 and self.gameOver == False:
            self.board[pos1,pos2]=mark 
            valid = True
            if self.checkBoard() >= 0:
                self.gameOver = True
            if self.gameOver == False:  #then move alt-agent
                validAlt = False
                currentState_alt = self.encodeBoardState_alt()
                while validAlt == False:
                    act = self.alt_agent.chooseActionEpsGreedy(currentState_alt)
                    (p1,p2) = self.decodeAction(act)
                    validAlt = self.moveAltAgent(p1,p2)
                    if validAlt == False:
                        self.alt_agent.invalidateLastAction()
            self.setReward()
            
        return valid
    def moveAltAgent(self,pos1,pos2):
        mark = -1
        valid=False
        if pos1>=0 and pos1 <3 and pos2>=0 and pos2 <3 and self.board[pos1,pos2]==0 and self.gameOver == False:
            self.board[pos1,pos2]=mark 
            valid = True           
        return valid
        
    def setReward(self):
        boardStatus = self.checkBoard() 
        self.status = boardStatus
        if boardStatus == 1:
            #alt agent won
            self.lastReward = -1
            self.gameOver = True
            self.terminal = True
        elif boardStatus == 2:
            #agent won
            self.lastReward = 1
            self.gameOver = True
            self.terminal = True
        elif boardStatus == 0:
            #tie
            self.lastReward = 0
            self.gameOver = True
            self.terminal = True
        else:
            #game not over
            self.lastReward = 0
            self.gameOver = False
            self.terminal = False
    def encodeBoardState(self):
        count=0
        state=0
        a=[0,1,2]
        for m in range(0,3):
            for n in range(0,3):                
                state+=a[int(self.board[m,n])]*(3**count)
                count+=1
        return state
    def encodeBoardState_alt(self):
        #this is for the alt agent only ( since it uses -1 rather than 1)
        count=0
        state=0
        a=[0,1,2]
        for m in range(0,3):
            for n in range(0,3):                
                state+=a[int(-self.board[m,n])]*(3**count)
                count+=1
        return state
    def encodeAction(self,pos1,pos2):
        action=np.ravel_multi_index((pos1,pos2),(3,3))
        return action
    def decodeAction(self,action):
        (p1,p2) = np.unravel_index(action,(3,3))
        return (p1,p2)    

