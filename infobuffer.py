# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 09:41:41 2024

@author: ANGGORO.BUDIONO
"""

import numpy as np

class InfoBuffer():
    #kusus episodic driver
    #untuk step driver dapat menggunakan replay buffer yang ada
    def __init__(self,env,n_iter):
        
        self.env=env
        self.info_shape=(1,3)
        self.n_episode=n_iter
        self.container= self.buffer()
        self.n=0
        
    def buffer(self):
        contain=np.zeros(shape=self.info_shape)
        return np.vstack([contain]*self.n_episode)
        
    def end_step(self): #bool
    
        timestep=self.env.current_time_step()
        return timestep.is_last()
        
    def __call__(self,item):
        
        if self.end_step():
            self.container[self.n]=item 
            self.n+=1 
    
    def get_all(self):
        x=self.container
        return {'cash': x[:,0],
                'lot': x[:,1],
                'aset': x[:,2]}
         
        
    