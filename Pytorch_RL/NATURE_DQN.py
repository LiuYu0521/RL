import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

BATCH_SIZE = 32
EPSILON = 0.9
GAMMA = 0.9
MEMORY_SIZE = 2000
TARGET_REPLACE_ITER = 100
N_HIDDEN = 20
LR = 0.001

class NET(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(NET, self).__init__()
        self.fc1 = nn.Linear(N_STATES, N_HIDDEN)
        self.out = nn.Linear(N_HIDDEN, N_ACTIONS)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class NATURE_DQN(object):
    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.batch_size = BATCH_SIZE
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.memory_size = MEMORY_SIZE
        self.learning_step_count = 0
        self.target_replace_iter = TARGET_REPLACE_ITER
        self.learning_rate = LR

        self.eval_net = NET(self.state_dim, self.action_dim)
        self.target_net = NET(self.state_dim, self.action_dim)

        self.memory = deque()

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

    def store_memory(self, s, a, r, s_, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[a] = 1
        self.memory.append((s, one_hot_action, r, s_, done))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()
        if len(self.memory) > self.batch_size:
            self.learn()

    def learn(self):
        if self.learning_step_count % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learning_step_count += 1

        minibatch = random.sample(self.memory, self.batch_size)
        batch_s = [i[0] for i in minibatch]
        batch_a = [i[1] for i in minibatch]
        batch_r = [i[2] for i in minibatch]
        batch_s_ = [i[3] for i in minibatch]

        batch_s = Variable(torch.FloatTensor(batch_s))
        batch_a = Variable(torch.FloatTensor(batch_a))
        batch_r = Variable(torch.FloatTensor(batch_r))
        batch_s_ = Variable(torch.FloatTensor(batch_s_))

        q_eval = (self.eval_net.forward(batch_s)*batch_a).sum(1)
        q_next = self.target_net.forward(batch_s_).detach()
        # q_next = self.eval_net.forward(batch_s_).detach()
        q_target = batch_r + self.gamma * q_next.max(1)[0]

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def choose_action_train(self, s):
        s = Variable(torch.FloatTensor(s))
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(s)
            action = np.argmax(actions_value.data.numpy())
        else:
            action = np.random.randint(0,self.action_dim)
        return action

    def choose_action_test(self, s):
        s = Variable(torch.FloatTensor(s))
        actions_value = self.eval_net.forward(s)
        action = np.argmax(actions_value.data.numpy())
        return action