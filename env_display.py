import gym
import cards_env


if __name__ == '__main__':
    env = gym.make(cards_env.register())
    obs = env.reset()

    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
