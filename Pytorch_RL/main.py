import gym

ENV_NAME = 'CartPole-v0'
EPISODE = 5000
STEP_LIMIT = 3000
TEST = 10

def main():
    env = gym.make(ENV_NAME)
    agent = NATURE_DQN(env)

    for episode in range(EPISODE):
        s = env.reset()

        for step in range(STEP_LIMIT):
            a = agent.choose_action_train(s)
            s_, r, done, info = agent.step(a)
            agent.store_memory(s, a, r, s_, done)
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

if __name__ == 'main':
    main()