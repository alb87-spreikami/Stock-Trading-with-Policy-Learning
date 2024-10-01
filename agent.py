# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 13:57:07 2024

agent.py sangat terbatas dalam mengakses variable state/info 
dari environment. 
Kelebihan menggunakan driver dan replay buffer dalam TF mode
sehingga memungkinkan menjalankan beberapa environment secara 
paralel. Sehingga training bisa lebih cepat.

@author: ANGGORO.BUDIONO
"""
#path loader suapaya tidak mengandalkan cache
import sys 

path='D:\\Matakul\\11. Financial Engineering\\rlfin'
sys.path.insert(0,path)

import datetime as dt
import tensorflow as tf
import pandas as pd
from tf_agents.utils import common
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.metrics.tf_metrics import NumberOfEpisodes, AverageReturnMetric
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver


from environmentRL import Env 

# =============================================================================
# INFLASI, sebagai parameter gamma
# Time Value of Money
# Inflasi 7% per tahun, 262 hari kerja dalam setahun
# =============================================================================

inflasi= 0.07 / 262
gamma=1-inflasi

# =============================================================================
# LOAD CHECKPOINT
# =============================================================================
#loading check point

direc='D:\\Matakul\\11. Financial Engineering\\RLFinEng\\chkptAdr'

env=Env()
env=TFPyEnvironment(env)

obs_spec=env.observation_spec()
act_spec=env.action_spec()
time_step_spec=env.time_step_spec()

n_iter=100

Q_network=QNetwork(input_tensor_spec=obs_spec,
                   action_spec=act_spec,
                   fc_layer_params=(64,32),
                   activation_fn=tf.keras.activations.relu)

global_step = tf.compat.v1.train.get_or_create_global_step()

epsilon = tf.compat.v1.train.polynomial_decay(learning_rate=0.9,
                                              global_step=global_step,
                                              decay_steps=n_iter,
                                              end_learning_rate=0.001)

agent=DdqnAgent(time_step_spec=time_step_spec,
                action_spec=act_spec,
                q_network=Q_network,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                epsilon_greedy=epsilon,
                n_step_update=8,
                gamma=gamma,
                target_update_period=int(10),
                train_step_counter=global_step)

data_spec=agent.collect_data_spec
replay_buffer=  TFUniformReplayBuffer(data_spec,
                                      batch_size=env.batch_size,
                                      max_length=100000)


check_point=common.Checkpointer(ckpt_dir=direc,
                                max_to_keep=1,
                                agent=agent,
                                policy=agent.policy,
                                replay_buffer=replay_buffer,
                                global_step=global_step)

check_point.initialize_or_restore()

# =============================================================================
# CONTINUE TRAINING
# =============================================================================
# melanjutkan training

if __name__=='__main__' :

    def train(enviro,agen,replaybuffer,n_iter):
    
        explore_policy=agen.collect_policy
        
    
        n_episode= NumberOfEpisodes()
        ave_return=AverageReturnMetric()
        
        replay_observer = [replaybuffer.add_batch,
                           n_episode,
                           ave_return]
        
        driver= DynamicEpisodeDriver(env=enviro,
                                     policy=explore_policy,
                                     observers=replay_observer)
        
        aset={}
        time_step=env.reset()
        for i in range(n_iter):
            t0=dt.datetime.now()
            driver.run(time_step)
            
            data_set=replaybuffer.as_dataset(sample_batch_size=64,num_steps=9)
            
            #training setiap iteration
            iterator=iter(data_set)
            for _ in range(1) :
                experience,_=next(iterator)
                agen.train(experience)
                
            #save check point setiap 100 episode
            if i==0 or (i%100)==0 or i==(n_iter-1):    
                check_point.save(global_step)
            
            #record aset setiap episode
            aset[i]=(n_episode.result().numpy(),
                     ave_return.result().numpy())
            
            t1=dt.datetime.now()
            delt=t1-t0
            
            print('learning iteration ke-',i+1,
                  'Dengan average return',ave_return.result().numpy())
            print('waktu yang dibutuhkan:{} second'.format(delt.seconds))
                
        aset_df= pd.DataFrame(aset)
        directory_xls='D:\\Matakul\\11. Financial Engineering\\RLFinEng'+'\\aset.xlsx'
        aset_df.to_excel(directory_xls)

    train(enviro=env,
          agen=agent,
          replaybuffer=replay_buffer,
          n_iter=n_iter
          )






