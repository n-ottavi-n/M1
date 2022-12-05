# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 10:33:34 2022

@author: notta
"""


import gym
import pygame
import numpy as np


env=gym.make("FrozenLake-v1", render_mode='human')

env.reset()

action=env.action_space.sample()
cont=env.step(action)



nb_iterations=100
threshold=10**(-20)
#threshold=0.0000001

gamma=1

global nb_states
nb_states=env.observation_space.n



'''
action 
0 LEFT
1 DOWN
2 RIGHT
3 UP
'''

def policy_iteration(): # ex2.3
    policy=np.random.randint(0, high=4, size=nb_states) #politique aleatoire
    policy_prec=policy.copy()
    table=compute_value_function(policy)
    for i in range(nb_iterations):
        table=compute_value_function(policy)
        policy=extract_policy(table)
        if np.array_equal(policy,policy_prec):
            break
        else:
            policy_prec=policy.copy()
    return policy


def compute_value_function(policy): # ex 2.1
    value_table=np.zeros(nb_states) #value table initialisee a 0
    for i in range(nb_iterations):
        v_pi=0
        value_table_prec=value_table.copy()
        for start_state in range(nb_states):
            v_pi=0
            action=policy[start_state]
            for end_state in range(nb_states):
                v_pi+=prob(end_state,action,start_state)*(reward(end_state,action,start_state)+gamma*value_table[end_state])
            value_table[start_state]=v_pi
        if(abs(np.sum(value_table)-np.sum(value_table_prec)))<=threshold:
            break
    return value_table

    

def extract_policy(value_table): #ex 1.2
    q_table=q_table=np.zeros((nb_states,4))
    policy=np.zeros(nb_states) 
    q=0
    for start_state in range(nb_states):
        for action in range(4):
            for end_state in range(nb_states):
                p=prob(end_state,action,start_state)
                r=reward(end_state,action,start_state)
                q=q+(p*(r+(gamma*value_table[end_state])))
            q_table[start_state,action]=q
            q=0   
    for state in range(nb_states):
        line=np.array(q_table[state])
        best_action_value=max(line)
        policy[state]=np.where(line==best_action_value)[0][0]
    return policy


def value_iteration(value_table):  #ex 1.1
    q_table=np.zeros((nb_states,4))
    value_table_prec=np.random.rand(nb_states)    
    iterations=0
    #while(np.allclose(value_table,value_table_prec,atol=0,rtol=threshold)==False and iterations<1000):
    while(abs(np.sum(value_table)-np.sum(value_table_prec))>=threshold and iterations<1000):  
        q=0
        iterations+=1 
        value_table_prec=value_table.copy()
        for start_state in range(nb_states):
            for action in range(4):
                for end_state in range(nb_states):
                    p=prob(end_state,action,start_state)
                    r=reward(end_state,action,start_state)
                    q=q+(p*(r+(gamma*value_table[end_state])))
                q_table[start_state,action]=q
                q=0
            value_table[start_state]=max(q_table[start_state])
    return value_table


def prob(s_prime,a,s): #-->probability of ending up at s_prime from s with action a 
    res=0
    p_list=env.P[s][a]
    for ligne in p_list:
        if ligne[1]==s_prime:
            res=ligne[0] # probability
    return res

def reward(s_prime,a,s): #-->reward of going from s to s_prime with action a
    res=0
    p_list=env.P[s][a]
    for ligne in p_list:
        if ligne[1]==s_prime:
            res=ligne[2] # reward
    return res



###### FOR Q ITERATTION ########
#rand_table=np.zeros(nb_states) 
#table=value_iteration(rand_table)
#final_policy=extract_policy(table)

###### FOR POLICY ITERATION #####
final_policy=policy_iteration()


n_episodes=100

score=0
for i in range(n_episodes):
    env.reset()
    time_step=0
    print("episode: ", i)
    state=0
    action=int(final_policy[state])
    state, r, done, info, probs=env.step(action)
    while not done:
        action=int(final_policy[state])
        state, r, done, info, probs=env.step(action)
        time_step+=1
        score+=r
    print(score)
env.close()
