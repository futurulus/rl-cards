# -*- coding: utf-8 -*-
import argparse
import gym
import numpy as np
from Queue import PriorityQueue

from stanza.monitoring import progress
from stanza.research import config
from stanza.research.learner import Learner

import cards_env
from cards_cache import all_transcripts

parser = config.get_options_parser()
parser.add_argument('--max_steps', type=int, default=1000,
                    help='The maximum number of actions to allow each agent in simulation.')
parser.add_argument('--render', type=config.boolean, default=False,
                    help='If True, display the enviroment in real time during testing.')


class CardsLearner(Learner):
    def __init__(self):
        self.get_options()

    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        pass

    @property
    def num_params(self):
        raise NotImplementedError

    def predict_and_score(self, eval_instances, random='ignored', verbosity=0):
        self.get_options()

        eval_instances = list(eval_instances)
        predictions = []
        scores = []

        env = gym.make(cards_env.register())

        if self.options.verbosity + verbosity >= 1:
            progress.start_task('Eval instance', len(eval_instances))

        for i, inst in enumerate(eval_instances):
            if self.options.verbosity + verbosity >= 1:
                progress.progress(i)

            total_reward = 0.

            trans = all_transcripts()[inst.input]
            env.reset()
            observation = env.configure(trans, verbosity=verbosity)
            info = None
            self.init_belief(env, observation)
            for step in range(self.options.max_steps):
                if self.options.render:
                    env.render()
                action = self.action(env, observation, info)
                prev_obs = observation
                observation, reward, done, info = env.step(action)
                self.update_belief(env, prev_obs, action, observation, reward, info)
                total_reward += reward
                if done:
                    break

            predictions.append('')
            scores.append(total_reward)

        env.close()

        if self.options.verbosity + verbosity >= 1:
            progress.end_task()

        return predictions, scores

    def action(self, env, observation, info, testing=True):
        raise NotImplementedError

    def init_belief(self, env, observation):
        raise NotImplementedError

    def update_belief(self, env, prev_obs, action, observation, reward, info):
        raise NotImplementedError

    def get_options(self):
        if not hasattr(self, 'options'):
            options = config.options()
            self.options = argparse.Namespace(**options.__dict__)


class RandomLearner(CardsLearner):
    '''
    An agent that simply takes random actions, and does not learn.
    '''
    @property
    def num_params(self):
        return 0

    def action(self, env, observation, info, testing=True):
        return env.action_space.sample()

    def init_belief(self, env, observation):
        pass

    def update_belief(self, env, prev_obs, action, observation, reward, info):
        pass


class SearcherLearner(CardsLearner):
    '''
    An agent that deterministically pathfinds its way to all squares with cards;
    if all squares have been explored and the game doesn't end, the agent then
    takes random actions.
    '''
    @property
    def num_params(self):
        return 0

    def action(self, env, observation, info, testing=True):
        if self.target_loc is None:
            return env.action_space.sample()
        elif not self.path:
            walls, cards, player, hand, floor, language = observation
            self.new_search(np.maximum(walls, self.invisible_walls), cards, player)

        if self.target_loc is None:
            return env.action_space.sample()
        else:
            action, self.prev_dest = self.path.pop()
            return cards_env.ACTIONS.index(action)

    def init_belief(self, env, observation):
        walls, cards, player, hand, floor, language = observation
        self.invisible_walls = np.zeros(walls.shape)
        self.explored = player
        self.new_search(walls, cards, player)

    def update_belief(self, env, prev_obs, action, observation, reward, info):
        walls, cards, player, hand, floor, language = observation
        self.explored = np.maximum(self.explored, player)

        _, _, prev_player, _, _, _ = prev_obs
        if self.target_loc is not None and (player == prev_player).all():
            # We're searching and stayed in one place. Must be an invisible
            # wall where we last tried to go.
            self.invisible_walls[self.prev_dest] = 1.
            self.new_search(np.maximum(walls, self.invisible_walls), cards, player)

    def new_search(self, walls, cards, player):
        to_search = np.minimum(cards, 1 - self.explored)
        if not to_search.any():
            to_search = np.minimum(1 - walls, 1 - self.explored)
        if not to_search.any():
            self.target_loc = None
            return

        paths = [(loc, pathfind(loc, player, walls))
                 for loc in all_locs(to_search)]
        path_len = lambda pair: float('inf') if pair[1] is None else len(pair[1])
        self.target_loc, self.path = min(paths, key=path_len)
        if self.path is None:
            self.target_loc = None
            return

        self.path.reverse()


def all_locs(board):
    '''
    >>> board = np.array([[1, 1], [0, 1]])
    >>> list(all_locs(board))
    [(0, 0), (0, 1), (1, 1)]
    '''
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            if board[r, c]:
                yield (r, c)


def pathfind(loc, player, walls):
    r'''
    >>> walls_str = '-----;' \
    ...             '- x--;' \
    ...             '-   -;' \
    ...             '- -p-;' \
    ...             '-----;'
    >>> walls = hyphens_to_walls(walls_str)
    >>> player = np.zeros(walls.shape)
    >>> player[3, 3] = 1.
    >>> pathfind((1, 2), player, walls)  # doctest: +NORMALIZE_WHITESPACE
    [('up', (2, 3)),
     ('left', (2, 2)),
     ('up', (1, 2))]
    '''
    player_loc = tuple(d[0] for d in np.where(player))
    backpointers = -np.ones(walls.shape + (2,), dtype=np.int)
    backpointers[walls.astype(np.bool), :] = 0
    backpointers[player_loc[0], player_loc[1], :] = player_loc
    queue = PriorityQueue()
    queue.put((manhattan_dist(player_loc, loc), player_loc))

    while not queue.empty():
        dist, curr = queue.get()
        if curr == loc:
            return follow_backpointers(curr, backpointers)

        steps = next_steps(curr, backpointers)
        for step in steps:
            backpointers[step[0], step[1], :] = curr
            queue.put((manhattan_dist(step, loc), step))

    # No path found
    return None


def follow_backpointers(curr, backpointers):
    path = []

    while True:
        back = tuple(backpointers[curr[0], curr[1], :])
        if back == curr:
            path.reverse()
            return path

        r, c = back
        if curr == (r, c + 1):
            action = 'right'
        elif curr == (r - 1, c):
            action = 'up'
        elif curr == (r, c - 1):
            action = 'left'
        elif curr == (r + 1, c):
            action = 'down'
        else:
            assert False, 'invalid step: {} -> {}'.format(back, curr)

        path.append((action, curr))
        curr = back


def manhattan_dist(loc_a, loc_b):
    '''
    >>> manhattan_dist((0, 1), (2, 0))
    3
    '''
    return abs(loc_a[0] - loc_b[0]) + abs(loc_a[1] - loc_b[1])


def next_steps(curr, backpointers):
    '''
    >>> bp = np.array([
    ...     [[-1, -1], [ 1,  1], [-1, -1]],
    ...     [[ 0,  0], [ 1,  1], [-1, -1]],
    ...     [[-1, -1], [-1, -1], [-1, -1]],
    ... ])
    >>> next_steps((1, 1), bp)  # doctest: +NORMALIZE_WHITESPACE
    [(1, 2),
     (2, 1)]
    '''
    r, c = curr
    steps = [
        (r, c + 1),
        (r - 1, c),
        (r, c - 1),
        (r + 1, c),
    ]
    return [(sr, sc) for (sr, sc) in steps
            if 0 <= sr < backpointers.shape[0] and
            0 <= sc < backpointers.shape[1] and
            backpointers[sr, sc, 0] < 0]


def hyphens_to_walls(walls_str):
    r'''
    >>> walls_str = '-----;' \
    ...             '- x--;' \
    ...             '-   -;' \
    ...             '- -p-;' \
    ...             '-----;'
    >>> hyphens_to_walls(walls_str)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  0.,  0.,  1.,  1.],
           [ 1.,  0.,  0.,  0.,  1.],
           [ 1.,  0.,  1.,  0.,  1.],
           [ 1.,  1.,  1.,  1.,  1.]])
    '''
    if walls_str.endswith(';'):
        walls_str = walls_str[:-1]
    return np.array([[1.0 if c == '-' else -1.0 if c == 'b' else 0.0
                      for c in row]
                     for row in walls_str.split(';')])
