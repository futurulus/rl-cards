import gym
import cards_env
import cards_cache


if __name__ == '__main__':
    env = gym.make(cards_env.register())

    transcript = 0

    for transcript in cards_cache.all_transcripts():
        env.reset()
        new_trans = transcript
        env.configure(new_trans, verbosity=2)
        for _ in range(100):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break
