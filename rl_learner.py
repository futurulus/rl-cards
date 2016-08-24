import gym
import numpy as np
import tensorflow as tf

from stanza.monitoring import progress
from stanza.research import config, iterators
from stanza.research.rng import get_rng

import cards_env
from baseline import CardsLearner
import tfutils
from tfutils import minimize_with_grad_clip

rng = get_rng()

parser = config.get_options_parser()
parser.add_argument('--pg_hidden_size', type=int, default=200,
                    help='The size of the neural network hidden layer for the Karpathy PG learner.')
parser.add_argument('--pg_batch_size', type=int, default=10,
                    help='The number of games to play in a batch between parameter updates.')
parser.add_argument('--pg_random_epsilon', type=float, default=0.1,
                    help='The probability of taking a uniform random action during training.')
parser.add_argument('--pg_grad_clip', type=float, default=5.0,
                    help='The maximum norm of a tensor gradient in training.')
parser.add_argument('--monitor_params', type=config.boolean, default=True,
                    help='If True, log histograms of parameters to Tensorboard.')
parser.add_argument('--monitor_grads', type=config.boolean, default=True,
                    help='If True, log histograms of gradients to Tensorboard.')
parser.add_argument('--monitor_activations', type=config.boolean, default=True,
                    help='If True, log histograms of gradients to Tensorboard.')
parser.add_argument('--detect_nans', type=config.boolean, default=True,
                    help='If True, error when a NaN is detected.')


class KarpathyPGLearner(CardsLearner):
    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        self.build_graph()
        env = gym.make(cards_env.register())

        # with gpu_session(self.graph) as sess:
        with tf.Session(graph=self.graph) as sess:
            self.session = sess
            self.init_params()

            batches = iterators.iter_batches(training_instances,
                                             self.options.pg_batch_size)
            num_batches = (len(training_instances) - 1) // self.options.pg_batch_size + 1

            if self.options.verbosity >= 1:
                progress.start_task('Batch', num_batches)

            try:
                for batch_num, batch in enumerate(batches):
                    if self.options.verbosity >= 1:
                        progress.progress(batch_num)
                    self.train_one_batch(list(batch), env, t=batch_num)
            except KeyboardInterrupt:
                self.summary_writer.flush()
                raise

            if self.options.verbosity >= 1:
                progress.end_task()

            self.session = None

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            walls = tf.placeholder(tf.float32, shape=(None,) + cards_env.MAX_BOARD_SIZE,
                                   name='walls')
            cards = tf.placeholder(tf.float32, shape=(None,) + cards_env.MAX_BOARD_SIZE,
                                   name='cards')
            player = tf.placeholder(tf.float32, shape=(None,) + cards_env.MAX_BOARD_SIZE,
                                    name='player')
            self.input_vars = [walls, cards, player]

            flattened = [tf.contrib.layers.flatten(t)
                         for t in (walls, cards, player)]
            combined_input = tf.concat(1, flattened, name='combined_input')

            fc = tf.contrib.layers.fully_connected
            hidden1 = fc(combined_input, trainable=True, num_outputs=self.options.pg_hidden_size)
            self.output = fc(hidden1, trainable=True, num_outputs=len(cards_env.ACTIONS))

            action = tf.placeholder(tf.int32, shape=(None,), name='action')
            reward = tf.placeholder(tf.float32, shape=(None,), name='reward')
            self.label_vars = [action, reward]

            reward_mean, reward_variance = tfutils.moments(reward)
            normalized = tf.nn.batch_normalization(reward, reward_mean, reward_variance,
                                                   scale=1.0, offset=0.0, variance_epsilon=1e-4)
            opt = tf.train.RMSPropOptimizer(learning_rate=0.1)
            logp = tf.nn.sparse_softmax_cross_entropy_with_logits(self.output, action,
                                                                  name='action_log_prob')
            dlogp = 1 - tf.exp(logp)
            loss = tf.reduce_mean(dlogp * normalized)
            tf.scalar_summary('loss', loss)
            var_list = tf.trainable_variables()
            print('Trainable variables:')
            for var in var_list:
                print(var.name)
            self.train_update = minimize_with_grad_clip(opt, self.options.pg_grad_clip,
                                                        loss, var_list=var_list)
            self.check_op = tf.add_check_numerics_ops()
            if self.options.monitor_activations:
                tfutils.add_summary_ops()
            self.summary_op = tf.merge_all_summaries()
            self.summary_writer = tf.train.SummaryWriter(self.options.run_dir, self.graph)

    def init_params(self):
        tf.initialize_all_variables().run()

    def train_one_batch(self, insts, env, t):
        inputs = []
        actions = []
        rewards = []

        if self.options.verbosity >= 1:
            progress.start_task('Instance', len(insts))

        for i, inst in enumerate(insts):
            if self.options.verbosity >= 1:
                progress.progress(i)
            env.reset()
            observation = env.configure(inst.input, verbosity=self.options.verbosity)
            info = None
            self.init_belief(env, observation)

            for step in range(self.options.max_steps):
                if self.options.render:
                    env.render()
                action = self.action(env, observation, info, testing=False)
                prev_obs = observation
                observation, reward, done, info = env.step(action)
                self.update_belief(env, prev_obs, action, observation, reward, info)
                if done:
                    break

            inputs.extend(self.inputs[:-1])
            actions.extend(self.actions)
            action_hist = np.bincount(self.actions, minlength=len(cards_env.ACTIONS)).tolist()
            print('Total reward: {}  {}'.format(sum(self.rewards), action_hist))
            rewards.extend([sum(self.rewards)] * len(self.rewards))

        if self.options.verbosity >= 1:
            progress.end_task()

        feed_dict = self.batch_inputs(inputs)
        for label, value in zip(self.label_vars, [np.array(actions), np.array(rewards)]):
            feed_dict[label] = value
        ops = [self.train_update, self.summary_op]
        if self.options.detect_nans:
            ops.append(self.check_op)
        results = self.session.run(ops, feed_dict=feed_dict)
        summary = results[1]
        self.summary_writer.add_summary(summary, t)
        print('Adding summary: {}'.format(t))

    @property
    def num_params(self):
        return 0

    def action(self, env, observation, info, testing=True):
        inputs = self.preprocess(observation)
        feed_dict = self.batch_inputs([inputs])
        dist = self.session.run(self.output, feed_dict=feed_dict)
        random_epsilon = 0.0 if testing else self.options.pg_random_epsilon
        action = sample(dist, random_epsilon=random_epsilon)[0]
        self.actions.append(action)
        return action

    def preprocess(self, observation):
        walls, cards, player, hand, floor, language = observation
        return walls, cards, player

    def batch_inputs(self, inputs):
        feed_dict = {}
        for i, input_var in enumerate(self.input_vars):
            feed_dict[input_var] = np.array([inp[i] for inp in inputs])
        return feed_dict

    def init_belief(self, env, observation):
        self.actions = []
        self.rewards = []
        self.inputs = [self.preprocess(observation)]

    def update_belief(self, env, prev_obs, action, observation, reward, info):
        self.rewards.append(reward)
        self.inputs.append(self.preprocess(observation))


def sample(a, temperature=1.0, random_epsilon=0.0, verbose=False):
    # helper function to sample an index from a log probability array
    a = np.array(a)
    if len(a.shape) < 1:
        raise ValueError('scalar is not a valid probability distribution')
    elif len(a.shape) == 1:
        # Cast to higher resolution to try to get high-precision normalization
        a = np.exp(a / temperature).astype(np.float64)
        a /= np.sum(a)
        if random_epsilon:
            a = random_epsilon * np.ones(a.shape, dtype=np.float64) / a.shape[0] + \
                (1 - random_epsilon) * a
            a /= np.sum(a)
            if verbose:
                print(a.tolist())
        return np.argmax(rng.multinomial(1, a, 1))
    else:
        return np.array([sample(s, temperature, random_epsilon, verbose=verbose) for s in a])
