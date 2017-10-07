import tensorflow as tf
import numpy as np
import pandas as pd
import random
from collections import  deque
import gym


ACTION_DIM = 2
N_HIDDEN = 20
ENV_NAME = 'LunarLander-v2'
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
STATE_DIM=4
MEMORY_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.9

class DQN(object):
    def __init__(self, env):#初始化
        self.memory = deque()
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.epsilon = INITIAL_EPSILON
        self.n_hidden = N_HIDDEN

        self.build_network()
        self.build_train()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def build_network(self):#创建神经网络
        self.state_input = tf.placeholder(tf.float32, [None, self.state_dim])
        l1 = tf.layers.dense(self.state_input, self.n_hidden, tf.nn.relu)
        self.Q_value = tf.layers.dense(l1, self.action_dim)

    def build_train(self):#创建训练方法
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
        self.Q_target = tf.placeholder(tf.float32, [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.Q_target - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def store_memory(self, observation, action, reward, observation_, done):#存储记忆
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.memory.append((observation, one_hot_action, reward, observation_, done))
        if len(self.memory) > MEMORY_SIZE:
            self.memory.popleft()
        if len(self.memory) > BATCH_SIZE:
            self.train()

    def train(self):#训练网络
        minibatch = random.sample(self.memory, BATCH_SIZE)
        state_batch = [memory_batch[0] for memory_batch in minibatch]
        action_batch = [memory_batch[1] for memory_batch in minibatch]
        reward_batch = [memory_batch[2] for memory_batch in minibatch]
        next_state_batch = [memory_batch[3] for memory_batch in minibatch]
        y_batch = []
        Q_value_batch = self.sess.run(self.Q_value, feed_dict={self.state_input:next_state_batch})
        for i in range(BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
        self.optimizer.run(feed_dict = {
            self.Q_target:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch
        })

    def train_choose_action(self, observation):#训练时动作选择
        Q_value = self.sess.run(self.Q_value, feed_dict={self.state_input: [observation]})
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

    def test_choose_action(self, observation):#测试时动作选择
        Q_value = self.sess.run(self.Q_value, feed_dict={self.state_input: [observation]})
        return np.argmax(Q_value)