import numpy as np
import tensorflow as tf
import random
from collections import deque

GAMMA = 0.9
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
MEMORY_SIZE = 10000
BATCH_SIZE = 32
N_HIDDEN = 20


class DQN(object):
    def __init__(self, env):
        self.memory = deque()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.epsilon = INITIAL_EPSILON
        self.n_hidden = N_HIDDEN

        # self.build_network()  # 创建神经网络
        # self.build_train()  # 创建训练方法
        self.build()

        self.sess = tf.InteractiveSession()
        writer = tf.summary.FileWriter("logs", self.sess.graph)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # merge_op = tf.summary.merge_all()


    # def build_network(self):  # 创建神经网络
    #     with tf.variable_scope('Input'):
    #         self.state_input = tf.placeholder(tf.float32, [None, self.state_dim], name='input')
    #     with tf.variable_scope('Net'):
    #         l1 = tf.layers.dense(self.state_input, self.n_hidden, tf.nn.relu, name='layer1')
    #         self.output = tf.layers.dense(l1, self.action_dim, name='output')
    #
    #         # tf.summary.histogram('h_out', self.l1)
    #         # tf.summary.histogram('pred', self.output)
    #
    # def build_train(self):  # 创建训练方法
    #     self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
    #     self.q_target = tf.placeholder(tf.float32, [None])
    #     q_action = tf.reduce_sum(tf.multiply(self.output, self.action_input), reduction_indices=1)
    #
    #     self.cost = tf.losses.mean_squared_error(self.q_target, q_action, scope='loss')
    #     # self.cost = tf.reduce_mean(tf.square(self.q_target - q_action))
    #     self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)
    #     # tf.summary.scalar('loss', self.cost)

    def build(self):
        pass

    def store_memory(self, state, action, reward, state_, done):  # 存储记忆
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.memory.append((state, one_hot_action, reward, state_, done))
        if len(self.memory) > MEMORY_SIZE:
            self.memory.popleft()
        if len(self.memory) > BATCH_SIZE:
            self.train()

    def train(self):  # 训练
        mini_batch = random.sample(self.memory, BATCH_SIZE)
        state_batch = [memory_batch[0] for memory_batch in mini_batch]
        action_batch = [memory_batch[1] for memory_batch in mini_batch]
        reward_batch = [memory_batch[2] for memory_batch in mini_batch]
        next_state_batch = [memory_batch[3] for memory_batch in mini_batch]
        q_target_batch = []
        q_value_next_batch = self.sess.run(self.output, feed_dict={self.state_input:next_state_batch})
        for i in range(BATCH_SIZE):
            done = mini_batch[i][4]
            if done:
                q_target_batch.append(reward_batch[i])
            else:
                q_target_batch.append(reward_batch[i] + GAMMA * np.max(q_value_next_batch[i]))
        # self.sess.run(self.optimizer, feed_dict={
        #                        self.state_input:state_batch,
        #                        self.action_input:action_batch,
        #                        self.q_target:q_target_batch
        #                    })
        self.optimizer.run(feed_dict =
                           {
                               self.state_input:state_batch,
                               self.action_input:action_batch,
                               self.q_target:q_target_batch
                           })

    def choose_action_train(self, state):  # 训练时动作选择
        q_value = self.sess.run(self.output, feed_dict={self.state_input:[state]})
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim-1)
        else:
            return np.argmax(q_value)

    def choose_action_test(self, state):  # 测试时动作选择
        q_value = self.sess.run(self.output, feed_dict={self.state_input:[state]})
        return np.argmax(q_value)
