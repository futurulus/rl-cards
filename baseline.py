# -*- coding: utf-8 -*-
import argparse
import gym
import numpy as np
from Queue import PriorityQueue

from stanza.monitoring import progress
from stanza.research import config, iterators
from stanza.research.learner import Learner

import cards_env
from helpers import profile

parser = config.get_options_parser()
parser.add_argument('--max_steps', type=int, default=1000,
                    help='The maximum number of actions to allow each agent in simulation.')
parser.add_argument('--render', type=config.boolean, default=False,
                    help='If True, display the enviroment in real time during testing.')


class CardsLearner(Learner):
    def __init__(self):
        self.get_options()

    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        raise NotImplementedError

    @property
    def num_params(self):
        raise NotImplementedError

    @profile
    def predict_and_score(self, eval_instances, random='ignored', verbosity=0):
        self.get_options()

        eval_instances = list(eval_instances)
        predictions = []
        scores = []

        env = gym.make(cards_env.register())

        batches = iterators.gen_batches(eval_instances, batch_size=cards_env.MAX_BATCH_SIZE)

        if self.options.verbosity + verbosity >= 1:
            progress.start_task('Eval batch', len(batches))

        for i, batch in enumerate(batches):
            batch = list(batch)
            if self.options.verbosity + verbosity >= 1:
                progress.progress(i)

            total_reward = np.zeros((len(batch),))
            done = np.zeros((len(batch),), dtype=np.bool)

            env.configure([inst.input for inst in batch], verbosity=verbosity)
            observation = env._get_obs()
            info = None
            self.init_belief(env, observation)

            if self.options.verbosity + verbosity >= 1:
                progress.start_task('Step', self.options.max_steps)

            for step in range(self.options.max_steps):
                if self.options.verbosity + verbosity >= 1:
                    progress.progress(step)
                if self.options.render:
                    env.render()
                action = self.action(env, observation, info)
                prev_obs = [np.copy(a) for a in observation]
                observation, reward, done_step, info = env.step(action)
                self.update_belief(env, prev_obs, action, observation, reward, done, info)
                done = np.bitwise_or(done, done_step[:len(batch)])
                total_reward += np.array(reward[:len(batch)]) * (1. - done)
                if done.all():
                    break

            if self.options.verbosity + verbosity >= 1:
                progress.end_task()

            predictions.extend([''] * len(batch))
            scores.extend(total_reward.tolist())

        env.close()

        if self.options.verbosity + verbosity >= 1:
            progress.end_task()

        return predictions, scores

    def action(self, env, observation, info, testing=True):
        raise NotImplementedError

    def init_belief(self, env, observation):
        raise NotImplementedError

    def update_belief(self, env, prev_obs, action, observation, reward, done, info):
        raise NotImplementedError

    def get_options(self):
        if not hasattr(self, 'options'):
            options = config.options()
            self.options = argparse.Namespace(**options.__dict__)

    def dump(self, prefix):
        '''
        :param prefix: The *path prefix* (a string, not a file-like object!)
                       of the model file to be written ('.pkl' will be added)
        '''
        with open(prefix + '.pkl', 'wb') as outfile:
            super(CardsLearner, self).dump(outfile)

    def load(self, filename):
        '''
        :param filename: The *path* (a string, not a file-like object!)
                         of the model file to be read
        '''
        with open(filename, 'rb') as infile:
            super(CardsLearner, self).load(infile)


class RandomLearner(CardsLearner):
    '''
    An agent that simply takes random actions, and does not learn.
    '''
    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        pass

    @property
    def num_params(self):
        return 0

    def action(self, env, observation, info, testing=True):
        return env.action_space.sample()

    def init_belief(self, env, observation):
        pass

    def update_belief(self, env, prev_obs, action, observation, reward, done, info):
        pass


class SearcherLearner(CardsLearner):
    '''
    An agent that deterministically pathfinds its way to all squares with cards;
    if all squares have been explored and the game doesn't end, the agent then
    takes random actions.
    '''
    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        pass

    @property
    def num_params(self):
        return 0

    @profile
    def action(self, env, observation, info, testing=True):
        actions = []
        for w in range(cards_env.MAX_BATCH_SIZE):
            if self.target_loc[w] is None:
                actions.append(env.action_space.sample()[0])
                continue
            elif not self.path[w]:
                walls, cards, player, hand, floor, language = observation[w * 6:(w + 1) * 6]
                self.new_search(w, np.maximum(walls, self.invisible_walls[w]), cards, player)

            if self.target_loc[w] is None:
                actions.append(env.action_space.sample()[0])
            else:
                action, self.prev_dest[w] = self.path[w].pop()
                actions.append(cards_env.ACTIONS.index(action))
        return actions

    def init_belief(self, env, observation):
        self.invisible_walls = []
        self.explored = []
        self.target_loc = []
        self.prev_dest = []
        self.path = []
        for w in range(cards_env.MAX_BATCH_SIZE):
            walls, cards, player, hand, floor, language = observation[w * 6:(w + 1) * 6]
            self.invisible_walls.append(np.zeros(walls.shape))
            self.explored.append(np.copy(player))
            self.target_loc.append(None)
            self.prev_dest.append(None)
            self.path.append(None)
            self.new_search(w, walls, cards, player)

    def update_belief(self, env, prev_obs, action, observation, reward, done, info):
        for w in range(cards_env.MAX_BATCH_SIZE):
            walls, cards, player, hand, floor, language = observation[w * 6:(w + 1) * 6]
            self.explored[w] = np.maximum(self.explored[w], player)

            prev_player = prev_obs[w * 6 + 2]
            if self.target_loc[w] is not None and (player == prev_player).all():
                # We're searching and stayed in one place. Must be an invisible
                # wall where we last tried to go.
                self.invisible_walls[w][self.prev_dest[w]] = 1.
                self.new_search(w, np.maximum(walls, self.invisible_walls[w]), cards, player)

    @profile
    def new_search(self, w, walls, cards, player):
        to_search = np.minimum(cards, 1 - self.explored[w])
        if not to_search.any():
            to_search = np.minimum(1 - walls, 1 - self.explored[w])
        if not to_search.any():
            self.target_loc[w] = None
            return

        paths = [(loc, pathfind(loc, player, walls))
                 for loc in all_locs(to_search)]
        path_len = lambda pair: float('inf') if pair[1] is None else len(pair[1])
        self.target_loc[w], self.path[w] = min(paths, key=path_len)
        if self.path[w] is None:
            self.target_loc[w] = None
            return

        self.path[w].reverse()


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


@profile
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


class OracleLearner(SearcherLearner):
    '''
    An agent that magically knows where the ace of spades is and tries to cut
    the quickest path to it. This agent cheats!
    '''
    def action(self, env, observation, info, testing=True):
        observation = self.remove_other_cards(observation)
        return super(OracleLearner, self).action(env, observation, info,
                                                 testing=testing)

    def init_belief(self, env, observation):
        self.ace_locs = [c2l[(0, 0)] for c2l in env.cards_to_loc]  # NOT FAIR!
        self.true_walls = [np.abs(w) for w in env.walls]
        observation = self.remove_other_cards(observation)
        return super(OracleLearner, self).init_belief(env, observation)

    def remove_other_cards(self, observation):
        observation = list(observation)
        for w, start in enumerate(range(0, len(observation), 6)):
            observation[start] = self.true_walls[w]
            cards = np.zeros(observation[start + 1].shape)
            cards[self.ace_locs[w]] = 1.
            observation[start + 1] = cards
        return tuple(observation)

    def update_belief(self, env, prev_obs, action, observation, reward, done, info):
        return super(OracleLearner, self).update_belief(env, prev_obs, action,
                                                        observation, reward, done, info)
