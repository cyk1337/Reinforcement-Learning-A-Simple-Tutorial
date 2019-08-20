#!/usr/bin/env python

#-*- encoding: utf-8

'''
_____.___._______________  __.____ __________    _________   ___ ___    _____  .___ 
\__  |   |\_   _____/    |/ _|    |   \      \   \_   ___ \ /   |   \  /  _  \ |   |
 /   |   | |    __)_|      < |    |   /   |   \  /    \  \//    ~    \/  /_\  \|   |
 \____   | |        \    |  \|    |  /    |    \ \     \___\    Y    /    |    \   |
 / ______|/_______  /____|__ \______/\____|__  /  \______  /\___|_  /\____|__  /___|
 \/               \/        \/               \/          \/       \/         \/     
 

@author: Yekun Chai
@license: Center on Research of Intelligent Systems and Engineering, IA, CAS
@contact: chaiyekun@gmail.com
@file: MountainCar Gym.py
@time: 3/22/19 10:45 AM
@desc：       
               
'''

import gym

from DQN import DeepQNetwork

env = gym.make("MountainCar-v0")
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n, n_features=env.observation_space.shape[0],
                  learning_rate=.001, e_greedy=.9, replace_target_iter=300, memory_size=3000,
                  e_greedy_increment=.0002)

total_steps = 0

for episode in range(10):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        position, velocity = observation_

        # the higher the better
        reward = abs(position - (-.5))

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()

        ep_r += reward
        if done:
            get = "|get" if observation_[0]>=env.unwrapped.goal_position else "| ---"
            print("Epi:", episode, get, "| Ep_r ", round(ep_r, 4), "| Epsilon: ", round(RL.epsilon, 2))

            break
        observation = observation_
        total_steps += 1

    RL.plot_cost()