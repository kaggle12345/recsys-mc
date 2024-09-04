import tensorflow as tf
import numpy as np
# from tensorflow.python.ops.gen_math_ops import Exp
from datetime import datetime
import os
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import time
import random
import sys 

from model.actor import Actor
from model.critic import Critic
from model.enviroment import StimulateEnv
from model.ddpg import DDPG 
from model.embedding import VideoGenreEmbedding, UserVideoEmbedding
from model.ou_noise import OUNoise

Path = ''

# PATH_USER_DICT = os.path.join(hub, "user_dict.npy")
# # PATH_EVAL_DATSET = os.path.join(hub, "eval_dict.npy")
# PATH_USER_HISTORY_LENS = os.path.join(hub, 'users_history_len_local.npy')
# PATH_DATA_NUMBER = os.path.join(hub, "data_number.npy")

# users_dict = np.load(PATH_USER_DICT,allow_pickle='TRUE').item()
# # eval_users_dict = np.load(PATH_EVAL_DATSET,allow_pickle='TRUE').item()
# data_number = np.load(PATH_DATA_NUMBER,allow_pickle='TRUE').item()
# users_history_lens = np.load(PATH_USER_HISTORY_LENS, allow_pickle='TRUE')

PATH_USER_DICT = os.path.join(Path, "dataset/user_dict.npy")
PATH_TRAIN_DATASET = os.path.join(Path, "dataset/train_dict.npy")
PATH_EVAL_DATSET = os.path.join(Path, "dataset/eval_dict.npy")
PATH_USER_HISTORY_LENS = os.path.join(Path, 'dataset/users_history_len_local.npy')
PATH_DICTIONARY = os.path.join(Path, "dataset/dictionary.npy")
PATH_DATA_NUMBER = os.path.join(Path, "dataset/data_number.npy")

users_dict = np.load(PATH_USER_DICT,allow_pickle='TRUE').item()
eval_users_dict = np.load(PATH_EVAL_DATSET,allow_pickle='TRUE').item()
train_users_dict = np.load(PATH_TRAIN_DATASET,allow_pickle='TRUE').item()
dictionary = np.load(PATH_DICTIONARY,allow_pickle='TRUE').item()
data_number = np.load(PATH_DATA_NUMBER,allow_pickle='TRUE').item()
users_history_lens = np.load(PATH_USER_HISTORY_LENS, allow_pickle='TRUE')


all_items = {data[0] for i, k in users_dict.items() for data in k}  ## list video toan tap data 

user_dataset = eval_users_dict
users_num = data_number['users_num']
items_num = data_number['items_num']

STATE_SIZE = 5
num_actions = 5
output_dim = 5

user_id = 10
users_history_lens = round(len(eval_users_dict[user_id]) * 0.6)
watched_videos = [video[0] for video in eval_users_dict[user_id]][:users_history_lens]
newest_watched_video = [video[0] for video in eval_users_dict[user_id]][users_history_lens:][0]
enviroment = StimulateEnv(user_id, newest_watched_video, eval_users_dict, users_history_lens,STATE_SIZE)
episodes=10000

#Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
# agent  = DDPG(enviroment, users_num, items_num, num_actions, STATE_SIZE, output_dim)  # output_dim là output của State_emebedding, để 1445 vì đầu vào của actor.evaluate_actor là (1445,400)
exploration_noise = OUNoise(num_actions)
counter=0
reward_per_episode = 0    
total_reward=0
steps = 20
#saving reward:
reward_st = np.array([0])
    

for i in range(0, episodes):
    print("==== Starting episode no:",i,"====","\n")
    user_id, watched_videos, done = enviroment.reset()

    # user_id = 4833
    users_history_lens = round(len(eval_users_dict[user_id]) * 0.6)
    watched_videos = [video[0] for video in eval_users_dict[user_id]][:users_history_lens]
    newest_watched_video = [video[0] for video in eval_users_dict[user_id]][users_history_lens:]
    newest_watched_video_start = [video[0] for video in eval_users_dict[user_id]][users_history_lens:][0]
    
    enviroment = StimulateEnv(user_id, newest_watched_video_start, eval_users_dict, users_history_lens,STATE_SIZE)
    agent  = DDPG(enviroment, users_num, items_num, num_actions, STATE_SIZE, output_dim)  # output_dim là output của State_emebedding, để 1445 vì đầu vào của actor.evaluate_actor là (1445,400)

    # users_history_lens =  round(len(user_dataset[user_id]) * 0.6)
    # watched_videos =  [data[0] for data in eval_users_dict[user_id]][:users_history_lens]
    
    steps = len(newest_watched_video)
    reward_per_episode = 0
    for t in range(0, steps):
        x = watched_videos[- STATE_SIZE:]
        ## change shape to fit evaluate_actor 
        state_value = tf.convert_to_tensor(x, dtype=tf.float32)
        state_value = tf.expand_dims(state_value, axis=0)
        action = agent.evaluate_actor(np.reshape(state_value,[1, num_actions]))
        noise = exploration_noise.noise()
        action = action[0] + noise #Select action according to current policy and exploration noise
        # print("Action at step", t ," :",action,"\n")
        
        recommended_item = agent.recommend_item(action, all_items, x, top_k= 5)
        next_items_ids_embs, reward, done, _= enviroment.step(recommended_item, newest_watched_video[t], x)
        state_value_next = tf.convert_to_tensor(next_items_ids_embs, dtype=tf.float32)
        state_value_next = tf.expand_dims(state_value_next, axis=0)
        state_value_next = np.reshape(state_value_next,[1, num_actions])
        #add s_t,s_t+1,action,reward to experience memory
        agent.add_experience(state_value, state_value_next, action, reward, done)
        #train critic and actor network
        if counter > 5: 
            agent.train()
        reward_per_episode+=reward[0]
        counter+=1
        #check if episode ends:
        if (done or (t == steps-1)):
            print('EPISODE: ',i,' Steps: ',t,' Total Reward: ',reward_per_episode)
            print("Printing reward to file")
            exploration_noise.reset() #reinitializing random noise for action exploration
            reward_st = np.append(reward_st,reward_per_episode)
            np.savetxt('episode_reward.txt',reward_st, newline="\n")
            print('\n\n')
            break
        watched_videos.append(newest_watched_video[t])
total_reward+=reward_per_episode            
print("Average reward per episode {}".format(total_reward / episodes)) 
