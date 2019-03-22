#!/usr/bin/env python

# -*- encoding: utf-8

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
@file: MountainCar.py
@time: 3/22/19 5:06 PM
@desc：       
               
'''

import gym
from PG import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = -2000

RENDER = False

env = gym.make("MountainCar-v0")
env.seed(1)
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(n_actions=env.action_space.n, n_features=env.observation_space.shape[0],
                    learning_rate=.02, reward_decay=.995, output_graph=True)

for episode in range(1000):
    observation = env.reset()

    while True:
        if RENDER: env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.ep_rs)
            if "running_reward" not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * .99 + ep_rs_sum * .01

            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True
            print("episode:", episode, " reward:", int(running_reward))

            vt = RL.learn()

            if episode == 30:
                plt.plot(vt)
                plt.xlabel("episode steps")
                plt.ylabel("normalized state-action value")
                plt.show()

            break
        observation = observation_
