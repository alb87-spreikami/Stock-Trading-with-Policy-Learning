# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 12:19:55 2024

py_agent menggunakan replaybuffer dan driver mode python, 
sehingga memudahkan dalam akses berbgai info (state) saat 
melakukan training.
Namun tidak dapat melakukan training dalam paralel environment.

@author: ANGGORO.BUDIONO
"""

#path loader suapaya tidak mengandalkan cache
import sys 

path='D:\\Matakul\\11. Financial Engineering\\rlfin'
sys.path.insert(0,path)

from environmentRL import Env
from infobuffer import InfoBuffer 

import datetime as dt
import tensorflow as tf
import pandas as pd
from tf_agents.utils import common
from tf_agents.networks.q_network import QNetwork
from tf_agents.specs.tensor_spec import to_array_spec
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.replay_buffers.py_uniform_replay_buffer import PyUniformReplayBuffer
from tf_agents.metrics.tf_metrics import NumberOfEpisodes, AverageReturnMetric
from tf_agents.drivers.py_driver import PyDriver
from tf_agents.environments.tf_py_environment import TFPyEnvironment

# =============================================================================
# INFLASI, sebagai parameter gamma
# Time Value of Money
# Inflasi 7% per tahun, 262 hari kerja dalam setahun
# =============================================================================

inflasi= 0.07 / 262
gamma=1-inflasi

# =============================================================================
# AGENT DEFINITION
# =============================================================================
#loading check point

direc='D:\\Matakul\\11. Financial Engineering\\RLFinEng\\chkptAdr_py'

#untuk menggunakan methode dari tf_agent, env harus TF
#replay buffer dan driver menggunakan py
#wrapping TF
env=Env()
env=TFPyEnvironment(env)

n_iter=100

obs_spec=env.observation_spec()
act_spec=env.action_spec()
time_step_spec=env.time_step_spec()

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

# =============================================================================
# REPLAY BUFFER & OBSERVERS
# =============================================================================
#loading check point

data_spec=to_array_spec(agent.collect_data_spec)
replay_buffer= PyUniformReplayBuffer(data_spec, 
                                     capacity=100000)
info_bufer=InfoBuffer(env, n_iter)

n_episode= NumberOfEpisodes()
ave_return=AverageReturnMetric()

replay_observer = [replay_buffer.add_batch,
                   n_episode,
                   ave_return] 

# =============================================================================
# LOADING CHECK POINT
# =============================================================================
#loading check point

check_point=common.Checkpointer(ckpt_dir=direc,
                                max_to_keep=1,
                                agent=agent,
                                policy=agent.policy,
                                replay_buffer=replay_buffer,
                                global_step=global_step)

check_point.initialize_or_restore()

# =============================================================================
# DRIVER & TRAIN
# =============================================================================

if __name__ == '__main__':
    
    explore_policy=agent.collect_policy

    driver= PyDriver(env,
                     policy=explore_policy,
                     observers=replay_observer,
                     info_observers=[info_bufer],
                     max_episodes=1)
    
    time_step=env.reset()
    for i in range(n_iter):
        
        t0=dt.datetime.now()  
   
        driver.run(time_step)
        
        data_set=replay_buffer.as_dataset(sample_batch_size=64,
                                         num_steps=9)
        #training setiap iteration
        iterator=iter(data_set)
        for _ in range(1) :
            experience=next(iterator)
            agent.train(experience)
            
        #save check point setiap 100 episode
        if i==0 or (i%100)==0 or i==(n_iter-1):    
            check_point.save(global_step)
            
        t1=dt.datetime.now()
        delt=t1-t0
        
        print('learning iteration ke-',i+1,
              'Dengan average return',ave_return.result().numpy())
        print('waktu yang dibutuhkan:{} second'.format(delt.seconds))
    
    aset_df= pd.DataFrame(info_bufer.get_all())
    directory_xls='D:\\Matakul\\11. Financial Engineering\\RLFinEng'+'\\aset_py.xlsx'
    aset_df.to_excel(directory_xls)

