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
@license: IA, CAS
@contact: chaiyekun@gmail.com
@file: PG.py
@time: 3/22/19 3:59 PM
@desc：       
               
'''

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(self, n_actions, n_features, learning_rate=.01, reward_decay=.95, output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope("inputs"):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="action_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # fc1
        fc1 = tf.layers.dense(inputs=self.tf_obs, units=10, activation=tf.nn.tanh,
                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=.3),
                              bias_initializer=tf.constant_initializer(.1), name="fc1")

        # fc2
        all_act = tf.layers.dense(inputs=fc1, units=self.n_actions, activation=None,
                                  kernel_initializer=tf.random_normal_initializer(0, .3),
                                  bias_initializer=tf.constant_initializer(.1),
                                  name="fc2")

        self.all_act_prob = tf.nn.softmax(all_act, name="act_prob")

        with tf.name_scope("loss"):
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)

            # or

            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob) * tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # shape-> (None, n_obs)
            self.tf_acts: np.array(self.ep_as),  # shape -> (None,)
            self.tf_vt: discounted_ep_rs_norm,  # shape -> (None,)
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode reward
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
