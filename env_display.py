import gym
import cards_env
import cards_cache
import world

from stanza.research.rng import get_rng

rng = get_rng()


if __name__ == '__main__':
    env = gym.make(cards_env.register())

    for transcript in cards_cache.all_transcripts():
        env.reset()
        env.configure([world.CardsWorld(transcript)], verbosity=2)
        for _ in range(100):
            env.render()
            dirs = ['right', 'up', 'left', 'down']
            action = dirs[rng.randint(0, len(dirs))]
            action_idx = cards_env.ACTIONS.index(action)
            observation, reward, done, info = env.step([action_idx])
            if done:
                break
