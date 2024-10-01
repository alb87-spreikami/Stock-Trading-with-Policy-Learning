# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:50:31 2024

Untuk simulasi dengan visualisasi. Menggunakan data pada environment
Agent yang telah di training di load, menggunakan policy dari hasil training
dilakukan simulasi kapan harus membeli, hold dan jual. 
Visualisasi dengan grafik.

@author: ANGGORO.BUDIONO
"""
#path loader suapaya tidak mengandalkan cache
import sys 
path='D:\\Matakul\\11. Financial Engineering\\rlfin'
sys.path.insert(0,path)

#import lstm_py_agent as lpa
#from environmentLSTM import EnvLstm
import py_agent as pa
from environmentRL import Env

from tf_agents.trajectories import time_step as ts
from tf_agents.environments import tf_py_environment

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

agent=pa.agent
policy=agent.policy


class SimEnv(Env):
    
    def __init__(self):
        super().__init__()
        
    def _reset(self):
        self.idx=0
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
    
#Karena dilakukan wrapping terhadap py_environment.dibuat Class baru (Sim: Simulasi)
#untuk mempermudah akses terhadapa variable2 py_env yg tidak bisa secara langsung diakses.

class Sim:
    
    def __init__(self,env): #env: SimEnv Class
        
        self.py_env = env 
        self.data_set=env.data_set
        self.data_date=env.data_date
        self.data_price=env.data_price 
        
    def feat(self,i):
        return self.data_set[i]
    
    def date(self,i):
        return self.data_date[i]
    
    def price(self,i):
        return self.data_price[i]
    
    def real_act(self,action,is_buy):
        action=action.numpy()[0]
        buyorhold=is_buy
        
        if action==1:
            act='hold'
        if (buyorhold and action==0):
            act='beli'
        if (buyorhold and action==2):
            act='hold'
        if (not buyorhold and action==0):
            act='hold'
        if (not buyorhold and action==2):
            act='jual'
            
        return act
    
    def cur_aset(self):
        return self.py_env.get_state()['total aset']
    
    def real_buy(self):
        return self.py_env._is_buyorhold
    
    @staticmethod
    def done(time_step):
        is_done=time_step.is_last()
        return is_done
    
    def plot(self,df):
        df['red_signal']=(df['action']=='beli')*df['price']
        df['green_signal']=(df['action']=='jual')*df['price']
        
        df_red_sign=df['red_signal']
        df_gre_sign=df['green_signal']

        sns.set_theme()
        fig, axs = plt.subplots(2, sharex=True)
        plt.figure(figsize=(50,20))
        
        axs[0].plot(df['price'],linewidth=0.5,color='r',label="Price")
        axs[1].plot(df['current aset'],linewidth=0.5,color='b',label="Aset")
        axs[0].plot(df_red_sign[df_red_sign>0],
                 color="r",marker="v",markersize=5,linestyle="None",label='beli')
        axs[0].plot(df_gre_sign[df_gre_sign>0],
                 color="g",marker="^",markersize=5,linestyle="None",label='jual')
        fig.legend()
        fig.show()
        
        

env=SimEnv()
sim=Sim(env)
tf_env=tf_py_environment.TFPyEnvironment(env)

if __name__=='__main__':
    
    time_step=tf_env.reset() 
    policy_state=policy.get_initial_state(tf_env.batch_size)
    i=0
    
    
    action_s=[]
    buy_s=[]
    real_act_s=[]
    current_aset_s=[]
    while not sim.done(time_step):
        current_aset=sim.cur_aset() #sebelum action
        current_aset_s.append(current_aset)
        is_buy=sim.real_buy() #sebelum action
        buy_s.append(is_buy)  
        
        policy_step=policy.action(time_step,policy_state)
        action=policy_step.action
        policy_state=policy_step.state
        time_step=tf_env.step(action)
        
        action_s.append(action.numpy()[0])
        real_act_s.append(sim.real_act(action,is_buy))
        i+=1
    
    df=pd.DataFrame({'date':sim.data_date,
                     'price': sim.data_price,
                    'action':real_act_s,
                    'current aset': current_aset_s},
                    columns=['date','price','action','current aset'])
    df.set_index('date',drop=True,inplace=True)
    df.to_excel('df.xlsx')
    
    sim.plot(df)
    

    

    
