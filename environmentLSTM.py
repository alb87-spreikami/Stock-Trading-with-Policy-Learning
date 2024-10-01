# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:14:29 2024

@author: ANGGORO.BUDIONO
"""

import technicalindicator as ti
from environmentRL import Env


import numpy as np
from tf_agents.specs import array_spec
from tf_agents.environments import tf_py_environment

# =============================================================================
# Loading Data
# Data tidak tersedian untuk recurent NN
# =============================================================================

def roll_window(data_set,data_date,data_price):
    data_set,data_date,data_price=ti.roll_window(data_set,
                                  data_date, 
                                  data_price,
                                  window=20)
    return data_set, data_date, data_price

# =============================================================================
# Environment Stock PyEnvironment 
# menjalankan dalam batch env, wrapper object menggunakan tf_py_environment
# EnvLstm digunakan untuk env agent menggunakan network recurrent atau LSTM
# =============================================================================

class EnvLstm(Env):
    
    def __init__(self):
        super().__init__()

        data=roll_window(self.data_set, 
                         self.data_date,
                         self.data_price)
        
        self.data_set=data[0]
        self.data_date=data[1]
        self.data_price=data[2]
        
        self._observation_spec=self.obsspec_func()

    def obsspec_func(self):
        shape=self.data_set.shape
        return array_spec.ArraySpec(shape=(1,shape[1],shape[2]+1), 
                        dtype=np.float64, name='observation')

        
    def feature(self,idx):
        feat=self.data_set[idx]    #data set pada index tertentu
        ind=feat.shape        #shape (tstep,feat_ind), size 2 dim
        tstep_ind=ind[0]      #int jumlah index tstep
        feat_ind=ind[1]       #int jumlah index feature
        boh=np.array([self._is_buyorhold]*tstep_ind).astype(dtype=np.float32)
        np_feature=np.zeros(shape=(len(feat),feat_ind+1))
        np_feature[:,:-1]=feat
        np_feature[:,-1]=boh
        return np_feature.reshape(1,np_feature.shape[0],
                                  np_feature.shape[1])

if __name__=='__main__':
    env=EnvLstm()
    env=tf_py_environment.TFPyEnvironment(env)
    