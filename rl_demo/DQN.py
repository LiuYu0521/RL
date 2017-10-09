import numpy as np
import tensorflow as tf
import random
from collections import deque

GAMMA = 0.9
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.01
MEMORY_SIZE = 5000
BATCH_SIZE = 32
N_HIDDEN = 20
REPLACE_TARGET_ITER = 300
LEARNING_RATE = 0.01
#
#
# class DQN(object):
#     def __init__(self, env, output_graph = False, Nature_DQN = False):
#         self.memory = deque()
#         self.state_dim = env.observation_space.shape[0]
#         self.action_dim = env.action_space.n
#         self.epsilon = INITIAL_EPSILON
#         self.n_hidden = N_HIDDEN
#         self.Nature_DQN = Nature_DQN
#         self.learn_step_count = 0
#
#         # self.build_network()  # 创建神经网络
#         # self.build_train()  # 创建训练方法
#         if self.Nature_DQN:
#             self.NATURE_DQN_build()
#             t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
#             e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
#             self.target_replace_op = [tf.assign(t,e) for t, e in zip(t_params, e_params)]
#         else:
#             self.build()
#
#         self.sess = tf.InteractiveSession()
#         if output_graph:
#             writer = tf.summary.FileWriter("logs", self.sess.graph)
#         init = tf.global_variables_initializer()
#         self.sess.run(init)
#
#         # merge_op = tf.summary.merge_all()
#
#
#     # def build_network(self):  # 创建神经网络
#     #     with tf.variable_scope('Input'):
#     #         self.state_input = tf.placeholder(tf.float32, [None, self.state_dim], name='input')
#     #     with tf.variable_scope('Net'):
#     #         l1 = tf.layers.dense(self.state_input, self.n_hidden, tf.nn.relu, name='layer1')
#     #         self.output = tf.layers.dense(l1, self.action_dim, name='output')
#     #
#     #         # tf.summary.histogram('h_out', self.l1)
#     #         # tf.summary.histogram('pred', self.output)
#     #
#     # def build_train(self):  # 创建训练方法
#     #     self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
#     #     self.q_target = tf.placeholder(tf.float32, [None])
#     #     q_action = tf.reduce_sum(tf.multiply(self.output, self.action_input), reduction_indices=1)
#     #
#     #     self.cost = tf.losses.mean_squared_error(self.q_target, q_action, scope='loss')
#     #     # self.cost = tf.reduce_mean(tf.square(self.q_target - q_action))
#     #     self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)
#     #     # tf.summary.scalar('loss', self.cost)
#
#     def build(self):
#         self.state_input = tf.placeholder(tf.float32, [None, self.state_dim], name='state_input')
#         with tf.variable_scope('Net'):
#             l1 = tf.layers.dense(self.state_input, self.n_hidden, tf.nn.relu, name = 'layer1')
#             self.output = tf.layers.dense(l1, self.action_dim, name='output')
#         self.action_input = tf.placeholder(tf.float32, [None, self.action_dim], name='action_input')
#         self.q_target = tf.placeholder(tf.float32, [None], name='q_target')
#         self.q_eval = tf.reduce_sum(tf.multiply(self.output, self.action_input), reduction_indices=1, name='q_action')
#         self.cost = tf.losses.mean_squared_error(self.q_target, self.q_eval, scope='loss')
#         self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
#
#     def NATURE_DQN_build(self):
#         self.state_input = tf.placeholder(tf.float32, [None, self.state_dim], name='state_input')
#         self.next_state_input = tf.placeholder(tf.float32, [None, self.state_dim], name='next_state_input')
#         self.action_input = tf.placeholder(tf.float32, [None, self.action_dim], name='action_input')
#
#         #build evaluate_net
#         with tf.variable_scope('eval_net'):
#             eval_l1 = tf.layers.dense(self.state_input, self.n_hidden, tf.nn.relu, name='eval_layer1')
#             self.eval_output = tf.layers.dense(eval_l1, self.action_dim, name='eval_output')
#         self.q_eval = tf.reduce_sum(tf.multiply(self.eval_output, self.action_input), reduction_indices=1, name='eval_q_action')
#
#         #build target_net
#         with tf.variable_scope('target_net'):
#             target_l1 = tf.layers.dense(self.next_state_input, self.n_hidden, tf.nn.relu, name='target_layer1')
#             self.target_output = tf.layers.dense(target_l1, self.action_dim, name='target_output')
#         self.q_target = tf.reduce_sum(tf.multiply(self.target_output, self.action_input), reduction_indices=1, name='target_q_action')
#
#         self.cost = tf.losses.mean_squared_error(self.q_target, self.q_eval, scope='loss')
#         self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
#
#     def store_memory(self, state, action, reward, state_, done):  # 存储记忆
#         one_hot_action = np.zeros(self.action_dim)
#         one_hot_action[action] = 1
#         self.memory.append((state, one_hot_action, reward, state_, done))
#         if len(self.memory) > MEMORY_SIZE:
#             self.memory.popleft()
#         if len(self.memory) > BATCH_SIZE:
#             self.learn()
#
#     def learn(self):  # 训练
#         mini_batch = random.sample(self.memory, BATCH_SIZE)
#         state_batch = [memory_batch[0] for memory_batch in mini_batch]
#         action_batch = [memory_batch[1] for memory_batch in mini_batch]
#         reward_batch = [memory_batch[2] for memory_batch in mini_batch]
#         next_state_batch = [memory_batch[3] for memory_batch in mini_batch]
#         q_target_batch = []
#         if self.Nature_DQN:
#             q_value_next_batch = self.sess.run(self.target_output, feed_dict={self.next_state_input:next_state_batch})
#         else:
#             q_value_next_batch = self.sess.run(self.eval_output, feed_dict={self.state_input:next_state_batch})
#         for i in range(BATCH_SIZE):
#             done = mini_batch[i][4]
#             if done:
#                 q_target_batch.append(reward_batch[i])
#             else:
#                 q_target_batch.append(reward_batch[i] + GAMMA * np.max(q_value_next_batch[i]))
#         # self.sess.run(self.optimizer, feed_dict={
#         #                        self.state_input:state_batch,
#         #                        self.action_input:action_batch,
#         #                        self.q_target:q_target_batch
#         #                    })
#         self.optimizer.run(feed_dict =
#                            {
#                                self.state_input:state_batch,
#                                self.action_input:action_batch,
#                                self.q_target:q_target_batch,
#                            })
#         self.learn_step_count += 1
#
#     def choose_action_train(self, state):  # 训练时动作选择
#         q_value = self.sess.run(self.eval_output, feed_dict={self.state_input:[state]})
#         if random.random() <= self.epsilon:
#             return random.randint(0, self.action_dim-1)
#         else:
#             return np.argmax(q_value)
#
#     def choose_action_test(self, state):  # 测试时动作选择
#         q_value = self.sess.run(self.eval_output, feed_dict={self.state_input:[state]})
#         return np.argmax(q_value)


#Morvan
# class DQN(object):
#     def __init__(self, env, output_graph = False):
#         self.state_dim = env.observation_space.shape[0]
#         self.action_dim = env.action_space.n
#         self.epsilon = INITIAL_EPSILON
#         self.n_hidden = N_HIDDEN
#         self.gamma = GAMMA
#         self.memory_size = MEMORY_SIZE
#         self.batch_size = BATCH_SIZE
#         self.learning_rate = LEARNING_RATE
#
#         self.learn_step_counter = 0
#         self.replace_targer_iter = REPLACE_TARGET_ITER
#
#         self.memory = deque()
#         self.build_net()
#
#         t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
#         e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
#         self.target_replace_op = [tf.assign(t,e) for t,e in zip(t_params, e_params)]
#
#         self.sess = tf.Session()
#
#         if output_graph:
#             writer = tf.summary.FileWriter("logs/", self.sess.graph)
#
#         self.sess.run(tf.global_variables_initializer())
#
#
#
#     def build_net(self):
#         self.state = tf.placeholder(tf.float32, [None, self.state_dim], name='state')
#         self.next_state = tf.placeholder(tf.float32, [None, self.state_dim], name='next_state')
#         self.reward = tf.placeholder(tf.float32, [None, ], name='reward')
#         self.action = tf.placeholder(tf.float32, [None, ], name='action')
#
#         #build evaluate net
#         with tf.variable_scope('eval_net'):
#             eval_layer1 = tf.layers.dense(self.state, self.n_hidden, tf.nn.relu, name='eval_layer1')
#             self.q_eval = tf.layers.dense(eval_layer1, self.action_dim, name='net_q_eval')
#
#         #build target net
#         with tf.variable_scope('target_net'):
#             target_layer1 = tf.layers.dense(self.next_state, self.n_hidden, tf.nn.relu, name='target_layer1')
#             self.q_next = tf.layers.dense(target_layer1, self.action_dim, name='net_q_target')
#
#         with tf.variable_scope('q_target'):
#             q_target = self.reward + self.gamma * tf.reduce_max(self.q_next, axis=1, name = 'next_state_qmax')
#             self.q_target = tf.stop_gradient(q_target)
#
#         with tf.variable_scope('q_eval'):
#             a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)
#             self.q_eval_wtr_a = tf.gather_nd(params=self.q_eval, indices=a_indices)
#
#         with tf.variable_scope('loss'):
#             self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wtr_a, name='TD_error'))
#
#         with tf.variable_scope('train'):
#             self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
#
#     def store_memory(self, state, action, reward, state_,):
#         self.memory.append((state, action, reward, state_,))
#         if len(self.memory) > self.memory_size:
#             self.memory.popleft()
#         if len(self.memory) > self.batch_size:
#             self.learn()
#
#     def learn(self):
#         if self.learn_step_counter % self.replace_targer_iter == 0:
#             self.sess.run(self.target_replace_op)
#         mini_batch = random.sample(self.memory, self.batch_size)
#         _,cost = self.sess.run(
#             [self.train_op, self.loss],
#             feed_dict={
#                 self.state:mini_batch[0],
#                 self.action:mini_batch[1],
#                 self.reward:mini_batch[2],
#                 self.next_state:mini_batch[3],
#             }
#         )
#         self.learn_step_counter += 1
#
#     def choose_action_train(self, state):
#         if random.random() <= self.epsilon:
#             action = random.randint(0, self.action_dim-1)
#         else:
#             action_value = self.sess.run(self.q_eval, feed_dict={self.state:state})
#             action = np.argmax(action_value)
#         return action
#
#
#     def choose_action_test(self, state):
#         action_value = self.sess.run(self.q_eval, feed_dict={self.state: state})
#         action = np.argmax(action_value)
#         return action

class DQN(object):
    def __init__(self, env, output_graph = False):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.epsilon = INITIAL_EPSILON
        self.n_hidden = N_HIDDEN
        self.gamma = GAMMA
        self.memory_size = MEMORY_SIZE
        self.batch_size = BATCH_SIZE
        self.learning_rate = LEARNING_RATE
        self.learning_step_count = 0
        self.replace_target_iter = REPLACE_TARGET_ITER

        self.memory = deque()

        self.build()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        self.target_replace_op = [tf.assign(t,e) for t,e in zip(t_params, e_params)]

        self.sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if output_graph:
            writer = tf.summary.FileWriter("logs", self.sess.graph)

    def build(self):
        self.state_input = tf.placeholder(tf.float32, [None, self.state_dim], name='state_input')
        self.next_state_input = tf.placeholder(tf.float32, [None, self.state_dim], name='next_state_input')
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim], name='action_input')
        self.reward_input = tf.placeholder(tf.float32, [None,], name='reward_input')
        self.done_input = tf.placeholder(tf.bool, [None,], name='done_input')

        #build evaluate net
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.state_input, self.n_hidden, tf.nn.relu, name='e_layer1')
            self.eval_output = tf.layers.dense(e1, self.action_dim, name='eval_output')

        #build target net
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.next_state_input, self.n_hidden, tf.nn.relu, name='t_layer1')
            self.target_output = tf.layers.dense(t1, self.action_dim, name='target_output')

        with tf.variable_scope('q_eval'):
            q_eval = tf.reduce_sum(tf.multiply(self.eval_output, self.action_input), reduction_indices=1)

        with tf.variable_scope('q_target'):
            if self.done_input is not None:
                q_target = self.reward_input
            else:
                q_target = self.reward_input + self.gamma * tf.reduce_max(self.target_output)
            # q_target_stop = tf.stop_gradient(q_target)

        with tf.variable_scope('loss'):
            self.loss = tf.losses.mean_squared_error(q_eval, q_target)

        with tf.variable_scope('train'):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss,
                                                               var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net'))

    def store_memory(self, state, action, reward, state_, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.memory.append((state, one_hot_action, reward, state_, done))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()
        if len(self.memory) > self.batch_size:
            self.learn()

    def learn(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        state_batch = [memory_batch[0] for memory_batch in mini_batch]
        action_batch = [memory_batch[1] for memory_batch in mini_batch]
        reward_batch = [memory_batch[2] for memory_batch in mini_batch]
        next_state_batch = [memory_batch[3] for memory_batch in mini_batch]
        done_batch = [memory_batch[4] for memory_batch in mini_batch]

        if self.learning_step_count % self.replace_target_iter == 0:
            # print(self.learning_step_count)
            self.sess.run(self.target_replace_op)

        self.sess.run(
            [self.optimizer, self.loss],
            feed_dict={
                self.state_input: state_batch,
                self.next_state_input: next_state_batch,
                self.action_input: action_batch,
                self.reward_input: reward_batch,
                self.done_input: done_batch
            }
        )
        # self.optimizer.run(feed_dict={
        #     self.state_input:state_batch,
        #     self.next_state_input:next_state_batch,
        #     self.action_input:action_batch,
        #     self.reward_input:reward_batch,
        #     self.done_input:done_batch
        # })

        self.learning_step_count += 1

    def choose_action_train(self, state):
        q_value = self.sess.run(self.eval_output, feed_dict={self.state_input:[state]})
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim-1)
        else:
            return np.argmax(q_value)

    def choose_action_test(self, state):
        q_value = self.sess.run(self.eval_output, feed_dict={self.state_input:[state]})
        return np.argmax(q_value)
