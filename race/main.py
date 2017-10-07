import numpy as np
import tensorflow as tf
import gym
import random
from  collections import deque
from RL import DQN

# ENV_NAME = 'LunarLander-v2'
ENV_NAME = 'CartPole-v0'
EPISODE = 5000
STEP_LIMIT = 3000
TEST = 10

def main():
    env = gym.make(ENV_NAME)
    agent = DQN(env)
    for episode in range(EPISODE):
        observation = env.reset()
        for step in range(STEP_LIMIT):
            # env.render()
            action = agent.train_choose_action(observation)
            # print(env.observation_space)
            # action = env.action_space.sample()
            # print(action)
            observation_, reward, done, info = env.step(action)
            agent.store_memory(observation, action, reward, observation_, done)
            observation = observation_
            if done:
                break
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP_LIMIT):
                    env.render()
                    action = agent.test_choose_action(state)
                    state,reward,done,_ = env.step(action)
                    total_reward = total_reward + reward
                    if done:
                        break
            average_reward = total_reward / TEST
            print('episode:', episode, 'Average Reward:', average_reward)
            if average_reward >= 200:
                break

if __name__ == '__main__':
    main()
