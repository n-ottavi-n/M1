# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 14:57:50 2022

@author: notta
"""
import gym
import pygame


env=gym.make("FrozenLake-v1", render_mode='human')

env.reset()

action=env.action_space.sample()
cont=env.step(action)

n_episodes=10
print(env.observation_space.n)
'''
for i in range(n_episodes):
    env.reset()
    time_step=0
    print("episode: ", i)
    while not env.step(action)[2]:
        action=env.action_space.sample()
        env.step(action)
        time_step+=1
        print("step: ", time_step)
        env.render()
        '''
env.close()