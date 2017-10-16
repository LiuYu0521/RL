import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random

BATCH_SIZE = 32
EPSILON = 0.9
GAMMA = 0.9
MEMORY_SIZE = 2000
TARGET_REPLACE_ITER = 100
N_HIDDEN = 20
LR = 0.01

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
        self.memory_counter = 0
        self.target_replace_iter = TARGET_REPLACE_ITER
        self.learning_rate = LR

        self.memory = np.zeros((self.memory_size, self.state_dim * 2 + 2))

        self.eval_net = NET(self.state_dim, self.action_dim)
        self.target_net = NET(self.state_dim, self.action_dim)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

    def store_memory(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        if self.memory_counter > self.batch_size:
            self.learn()

    def learn(self):
        if self.learning_step_count % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learning_step_count += 1

        sample_index = np.random.choice(self.memory_size, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:,:self.state_dim]))
        b_a = Variable(torch.LongTensor(b_memory[:,self.state_dim:self.state_dim+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:,self.state_dim+1:self.state_dim+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:,-self.state_dim:]))

        q_eval = self.eval_net(b_s).gather(1,b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma*q_next.max(1)[0].view(self.batch_size,1)

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