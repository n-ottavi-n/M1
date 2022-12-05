# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 16:29:31 2022

@author: notta
"""

import gym
import pygame


env=gym.make("CartPole-v0", render_mode='human')

env.reset()

num_episodes=100
num_timesteps=50
ret=0




for i in range(num_episodes):
    env.reset()
    for j in range(num_timesteps):
        action=env.action_space.sample()
        reward=env.step(action)[1]
        env.render()
        ret+=reward
    if i%10==0:
        print("Episode: ", i, " Return: ", ret)
    ret=0

env.close()
