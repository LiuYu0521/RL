import gym
import numpy as np
ENV_NAME = 'SpaceInvaders-v0'
EPISODE = 5000
STEP_LIMIT = 3000

def main():
    env = gym.make(ENV_NAME)
    obervation = env.reset()
    env.render()
    obervation_num = env.observation_space.shape[0]
    print(obervation_num)
    # print(env.observation_space.low)
    # print(env.observation_space.high)
    action = env.action_space.sample()
    print(env.action_space.n)
    # obervation_, reward, done, _ = env.step(action)

if __name__ == '__main__':
    main()