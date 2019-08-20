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
@file: openAI_gym.py
@time: 3/22/19 10:24 AM
@desc：       
               
'''

from DQN import DeepQNetwork
import gym

env = gym.make("CartPole-v0")
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],  # (4,)[0]
                  learning_rate=.01,
                  e_greedy=.9,
                  replace_target_iter=100,
                  memory_size=2000,
                  e_greedy_increment=.001,
                  )

total_steps = 0

for episode in range(1000):
    observation = env.reset()

    ep_r = 0

    while True:
        env.render()

        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - .8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - .5
        reward = r1 + r2

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward

        if total_steps > 1000:
            RL.learn()

        if done:
            print("episode: ", episode, "ep_r",round(ep_r, 2),
                  " epsilon:", round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()