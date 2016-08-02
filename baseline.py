# -*- coding: utf-8 -*-
import argparse
import gym

from stanza.monitoring import progress
from stanza.research import config
from stanza.research.learner import Learner

import cards_env
from cards_cache import all_transcripts

parser = config.get_options_parser()
parser.add_argument('--max_steps', type=int, default=1000,
                    help='The maximum number of actions to allow each agent in simulation.')


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
            for step in range(self.options.max_steps):
                action = self.action(env, observation, info)
                observation, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break

            predictions.append('')
            scores.append(total_reward)

        if self.options.verbosity + verbosity >= 1:
            progress.end_task()

        return predictions, scores

    def action(self, env, observation, info, testing=True):
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
