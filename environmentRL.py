# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:31:56 2024

@author: ANGGORO.BUDIONO
"""

import copy
import numpy as np
import datetime as dt 
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec 
from tf_agents.trajectories import time_step as ts 


from technicalindicator import TechnicalData as td


# =============================================================================
# Loading Data
# Data tidak tersedian untuk recurent NN
# =============================================================================
START_DATE=dt.date(2020,1,1)
TICKER=str(input('Silakan Masukan ticker saham....'))

data=td(s_date=START_DATE,
        ticker=TICKER)

# =============================================================================
# Environment Stock PyEnvironment 
# menjalankan dalam batch env, wrapper object menggunakan tf_py_environment
# =============================================================================
  
class Env(py_environment.PyEnvironment):
    
    def __init__(self):
        
        self.data_set=data()
        self.data_date=data.date
        self.data_price=data.price
        
        self._action_spec= array_spec.BoundedArraySpec(shape=(), 
                        dtype=np.int32, minimum=0, maximum=2, name='action')
        
        n_length=self.data_set.shape[1] + 1
        self._observation_spec=array_spec.ArraySpec(shape=(1,n_length), 
                        dtype=np.float64, name='observation')
        
        
        self._episode_ended=False 
        
        
# =============================================================================
#     def obsspec_func(self):
#         n_length=self.data_set.shape[1] + 1
#         return array_spec.ArraySpec(shape=(1,n_length), 
#                         dtype=np.float64, name='observation')
# =============================================================================
    
    def getinfo_spec(self):
        #cash,lot saham, total aset
        return array_spec.ArraySpec(shape=(1,3),dtype=np.int32,
                                    name='get_info')

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def time_step_spec(self):
        return ts.time_step_spec(self.observation_spec(),
                                 self.reward_spec())
   
    def feature(self,idx):
        feat=self.data_set[idx]
        boh=np.array(self._is_buyorhold).astype(dtype=np.float32).reshape(-1,)
        np_feature=np.zeros(shape=(len(feat)+len(boh),))
        np_feature[:-1]=feat
        np_feature[-1]=boh
        return np_feature.reshape((1,-1))
    
    ####################################### 
    # FORMULASI LANGKAH (STEP) |
    #######################################

    def _reset(self):
        self.idx=np.random.randint(low=0,high=(len(self.data_date)-5)) 
        #supaya tidak terlalu mepet ujung data
        self._episode_ended=False
        self._is_buyorhold=True
        self._observation= self.feature(self.idx)
        self._date=self.data_date[self.idx]
        self._cash=100000000 #modal awal 100juta
        self._lot=0 #per seratus lembar
        self._price=self.data_price[self.idx] #harga saham terbaru
        self._aset=self.total_aset()
        self._cost=0
        
        
        return ts.restart(self._observation,
                          reward_spec=self.reward_spec())
    
    def total_aset(self):
        return self._lot*100*self._price + self._cash

    
    def _step(self,action):
        #action : 0 -->buy, 1 --> hold, 2 --> sell
        
        if self._episode_ended:
            return self.reset()  
        else:
            return self.move(action)
            
       
        
    def move(self,action):
        
        if self.idx==(len(self.data_date)-1) or self._aset <= -10000000: 
            #kurang dari  -10juta, episode end
            self._episode_ended=True
            return ts.termination(self._observation,
                                 reward=0)
        
        else:
            self.idx+=1
            self._episode_ended=False
            self._observation= self.feature(self.idx)
            self._date=self.data_date[self.idx]
            self._price=self.data_price[self.idx]
            
            if action==0 and self._is_buyorhold==True:
                return self.move_buy()
                
            elif action==0 and self._is_buyorhold==False:
                return self.move_hold()
                
            elif action==1:
                return self.move_hold()
                
            elif action==2 and self._is_buyorhold==False:
                return self.move_sell() 
                
            elif action==2 and self._is_buyorhold==True:
                return self.move_hold()
                
    def move_buy(self):
 
        self._lot=self._cash//(self._price*100)
        self._cash-=(self._lot*100)*self._price
        self._aset=self.total_aset()
        self._cost=self._lot*100*self._price
        self._is_buyorhold=False
        return ts.transition(self._observation,
                             reward=0, discount=1.0)
    
    def move_hold(self):
        self._aset=self.total_aset()
        return ts.transition(self._observation,
                             reward=0, discount=1.0)
    
    def move_sell(self):
        
        self._cash+=(self._lot*100)*self._price
        reward_calc= ((self._lot*100*self._price) - self._cost)/1000000
        #dibagi 1 juta untuk kestabilan proses training
        self._lot=0
        self._aset=self.total_aset()
        self._cost=0
        self._is_buyorhold=True

        return ts.transition(self._observation,
                             reward=reward_calc, discount=1.0)
    @property
    def item (self):
        #merangkum variable kondisi terkini dari saham
        #namun hanya bisa digunakan dalam environement mode python
        return {'lot':int(self._lot),
                'price':int(self._price),
                'cash':int(self._cash),
                'aset':int(self._aset),
                'cost':int(self._cost),
                'date':self._date,
                'is_buyorhold':self._is_buyorhold}
    
    def get_info(self):
        #dapat diakses per step melalui mode batch env (TFEnvironment),
        #asalkan driver per step.
        #dapat diakses per episode mode eiger execution, dan dapat dimasukan dalam buffer
        #karena dimasukan dalam methode pydriver,
        #format numpy array cash,lot saham, total aset
        return np.array([self._cash,self._lot,int(self._aset)])

    def get_state(self):
        #merangkum variable kondisi terkini dari saham
        #tidak dapat diacces pada wrapping environment
        #gunaka pyenvironment dengan driver per step untuk mendapatkan 
        #data secara lengkap.
        return {'cash': copy.deepcopy(self._cash),
                'lot saham': copy.deepcopy(self._lot),
                'total aset': copy.deepcopy(self._aset)}
    
    @property
    def info(self):
        # memperoleh gambaran terkini dari saham
        # dengan format tulisan lebih mudah terbaca
        # tidak dapat diacces pada wrapping environment
        print('','cash:',"Rp {:,.2f}".format(self._cash),'\n',
              'jumlah stock (dalam lot):',int(self._lot),'\n',
              'total aset:',"Rp {:,.2f}".format(self._aset))
        

        
if __name__=='__main__':
    env=Env()
    env=tf_py_environment.TFPyEnvironment(env)              
            
        
    
    
        
        
        
        
        
        
    
    