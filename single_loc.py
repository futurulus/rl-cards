import gym
import numpy as np

import cards_env
from world import CardsWorld
from cards_cache import all_transcripts


def print_available_locs():
    cache = {}
    for trans in all_transcripts():
        world = CardsWorld(trans)
        key = str(world.walls)
        if key not in cache:
            cache[key] = world

    locs = get_available_locs(cache)
    for loc in locs:
        print(loc)

    env = gym.make(cards_env.register())
    for key, world in cache.iteritems():
        display_world(world, locs, env)


def get_available_locs(cache):
    max_walls = None
    for world in cache.itervalues():
        if max_walls is None:
            max_walls = np.abs(world.walls)
        else:
            max_walls = np.maximum(max_walls, np.abs(world.walls))

    return [(r, c) for r in range(max_walls.shape[0]) for c in range(max_walls.shape[1])
            if max_walls[r, c] == 0.0]


def display_world(world, locs, env):
    # show invisible walls as normal walls
    # and show available locs as invisible walls!
    locs_arr = locs_to_array(locs, world.walls)
    world.walls = (np.abs(np.array(world.walls)) - locs_arr).tolist()
    env.reset()
    env.configure(world, verbosity=0)
    for _ in range(100):
        env.render()


def locs_to_array(locs, walls):
    '''
    >>> locs_to_array([(0, 0), (1, 0)], np.array([[0, 1], [0, 1]]))
    array([[ 1.,  0.],
           [ 1.,  0.]])
    '''
    arr = np.zeros(np.array(walls).shape)
    for loc in locs:
        arr[loc] = 1.0
    return arr


if __name__ == '__main__':
    print_available_locs()
