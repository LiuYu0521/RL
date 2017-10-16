import gym
from NATURE_DQN import NATURE_DQN

ENV_NAME = 'CartPole-v0'
EPISODE = 5000
STEP_LIMIT = 500
TEST = 10

def main():
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    agent = NATURE_DQN(env)

    for episode in range(EPISODE):
        s = env.reset()

        for step in range(STEP_LIMIT):
            a = agent.choose_action_train(s)
            s_, r, done, info = env.step(a)

            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1+r2

            agent.store_memory(s, a, r, s_)
            s = s_
            if done:
                break

        if episode % 100 == 0:
            total_reward = 0

            for i in range(TEST):
                s = env.reset()

                for j in range(STEP_LIMIT):
                    env.render()
                    a = agent.choose_action_test(s)
                    s, r, done, info = env.step(a)
                    total_reward = total_reward + r
                    if done:
                        break

            average_reward = total_reward / TEST
            print('episode: ', episode, 'Average Reward: ', average_reward)

if __name__ == '__main__':
    main()