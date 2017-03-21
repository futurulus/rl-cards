import gym
import numpy as np
import tensorflow as tf

from stanza.monitoring import progress
from stanza.research import config, iterators, learner
from stanza.research.rng import get_rng

from world import cards
import cards_env
from baseline import CardsLearner
import datasets
import tfutils
from tfutils import minimize_with_grad_clip
from helpers import profile
import vectorizers

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
parser.add_argument('--pg_train_epochs', type=int, default=1,
                    help='Number of times to pass through the data in training.')
parser.add_argument('--pg_normalize', type=config.boolean, default=True,
                    help='If True, apply batch normalization to total rewards to reduce gradients.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Fraction of dimensions to drop in dropout (1 - keep_prob). '
                         'Use 0.0 to disable.')
parser.add_argument('--move_only', type=config.boolean, default=False,
                    help='If True, restrict actions to move actions (left, right, up, down).')
parser.add_argument('--bias_only', type=config.boolean, default=False,
                    help='If True, ignore input and only learn overall preference for actions.')
parser.add_argument('--monitor_params', type=config.boolean, default=True,
                    help='If True, log histograms of parameters to Tensorboard.')
parser.add_argument('--monitor_grads', type=config.boolean, default=True,
                    help='If True, log histograms of gradients to Tensorboard.')
parser.add_argument('--monitor_activations', type=config.boolean, default=True,
                    help='If True, log histograms of gradients to Tensorboard.')
parser.add_argument('--detect_nans', type=config.boolean, default=True,
                    help='If True, error when a NaN is detected.')


class Unpicklable(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '<%s removed in pickling>' % (self.name,)


class TensorflowLearner(learner.Learner):
    @property
    def num_params(self):
        total = 0
        with self.graph.as_default():
            for var in tf.trainable_variables():
                total += np.prod(var.eval(self.session).shape)
        return total

    @property
    def use_dropout(self):
        return True

    def init_params(self):
        self.step = 0
        with self.graph.as_default():
            tf.global_variables_initializer().run(session=self.session)

    def build_graph(self):
        if hasattr(self, 'graph'):
            return

        self.graph = tf.Graph()
        with self.graph.as_default():
            if self.use_dropout:
                self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
            self.input_vars, self.label_vars, self.train_op, self.predict_op = self.get_layers()
            self.check_op = tfutils.add_check_numerics_ops()
            if self.options.monitor_activations:
                tfutils.add_summary_ops()
            self.summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.options.run_dir, self.graph)
            self.saver = tf.train.Saver()
            self.run_metadata = tf.RunMetadata()
        # gpu_session(self.graph)
        config = tf.ConfigProto(device_count={'GPU': 0})
        self.session = tf.Session(graph=self.graph, config=config)

    def run_train(self, feed_dict):
        self.step += 1
        ops = [self.train_op]
        if self.summary_op is not None:
            ops.append(self.summary_op)
        if self.options.detect_nans:
            ops.append(self.check_op)
        if self.use_dropout:
            feed_dict = dict(feed_dict)
            feed_dict[self.dropout_keep_prob] = 1.0 - self.options.dropout
        results = self.session.run(ops, feed_dict=feed_dict)
        if self.summary_op is not None:
            summary = results[1]
            self.summary_writer.add_summary(summary, self.step)

    def run_predict(self, feed_dict):
        if self.use_dropout:
            feed_dict = dict(feed_dict)
            feed_dict[self.dropout_keep_prob] = 1.0
        return self.session.run(self.predict_op, feed_dict)

    def get_layers(self):
        raise NotImplementedError
        return 'input_vars', 'label_vars', 'train_op', 'predict_op'

    def dump(self, prefix):
        '''
        :param outfile: The *path prefix* (a string, not a file-like object!)
                        of the model file to be written
        '''
        self.build_graph()
        self.saver.save(self.session, prefix, global_step=0)
        with open(prefix + '.pkl', 'wb') as outfile:
            learner.Learner.dump(self, outfile)

    def load(self, filename):
        '''
        :param outfile: The *path* (a string, not a file-like object!)
                        of the model file to be read
        '''
        with open(filename + '.pkl', 'rb') as infile:
            super(TensorflowLearner, self).load(infile)
        del self.graph
        self.build_graph()
        if not hasattr(self, 'session'):
            self.session = tf.Session(graph=self.graph)
        self.saver.restore(self.session, filename + '-0')

    def __getstate__(self):
        state = dict(self.__dict__)
        for k in ['input_vars', 'label_vars', 'train_op', 'predict_op', 'check_op',
                  'dropout_keep_prob', 'session', 'graph',
                  'saver', 'summary_op', 'run_metadata', 'summary_writer']:
            if k in state:
                state[k] = Unpicklable(k)
        return state


class KarpathyPGLearner(TensorflowLearner, CardsLearner):
    @profile
    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        self.build_graph()
        env = gym.make(cards_env.register())

        self.init_params()

        if self.options.verbosity >= 1:
            progress.start_task('Epoch', self.options.pg_train_epochs)

        for epoch in range(self.options.pg_train_epochs):
            if self.options.verbosity >= 1:
                progress.progress(epoch)

            batches = iterators.iter_batches(training_instances,
                                             self.options.pg_batch_size)
            num_batches = (len(training_instances) - 1) // self.options.pg_batch_size + 1

            if self.options.verbosity >= 1:
                progress.start_task('Batch', num_batches)

            try:
                for batch_num, batch in enumerate(batches):
                    if self.options.verbosity >= 1:
                        progress.progress(batch_num)
                    step = epoch * num_batches + batch_num
                    self.train_one_batch(list(batch), env, t=step)
                    if step % 10 == 0:
                        check_prefix = config.get_file_path('checkpoint')
                        self.saver.save(self.session, check_prefix, global_step=step)
            except KeyboardInterrupt:
                self.summary_writer.flush()
                raise

            if self.options.verbosity >= 1:
                progress.end_task()

        if self.options.verbosity >= 1:
            progress.end_task()

    def get_layers(self):
        input_vars, predict_op = self.get_predict_op()

        action = tf.placeholder(tf.int32, shape=(None,), name='action')
        reward = tf.placeholder(tf.float32, shape=(None,), name='reward')
        credit = tf.placeholder(tf.float32, shape=(None,), name='credit')
        label_vars = [action, reward, credit]

        if self.options.pg_normalize:
            reward_mean, reward_variance = tfutils.moments(reward)
            normalized = tf.nn.batch_normalization(reward, reward_mean, reward_variance,
                                                   scale=1.0, offset=0.0, variance_epsilon=1e-4)
        else:
            normalized = reward
        opt = tf.train.RMSPropOptimizer(learning_rate=0.1)
        logp = tf.neg(tf.nn.sparse_softmax_cross_entropy_with_logits(predict_op, action),
                      name='action_log_prob')
        signal = tf.mul(logp, normalized * credit, name='signal')
        signal_down = tf.reduce_sum(tf.slice(tf.reshape(signal, [-1, 10]),
                                             [0, 4], [-1, 1]),
                                    [0], name='signal_down')
        print_node = tf.Print(signal, [signal_down], message='signal_down: ', summarize=10)
        with tf.control_dependencies([print_node]):
            signal = tf.identity(signal)
        loss = tf.reduce_mean(-signal, name='loss')
        var_list = tf.trainable_variables()
        print('Trainable variables:')
        for var in var_list:
            print(var.name)
        train_op = minimize_with_grad_clip(opt, self.options.pg_grad_clip,
                                           loss, var_list=var_list)

        return input_vars, label_vars, train_op, predict_op

    def get_predict_op(self):
        walls = tf.placeholder(tf.float32, shape=(None,) + cards_env.MAX_BOARD_SIZE,
                               name='walls')
        cards = tf.placeholder(tf.float32, shape=(None,) + cards_env.MAX_BOARD_SIZE,
                               name='cards')
        player = tf.placeholder(tf.float32, shape=(None,) + cards_env.MAX_BOARD_SIZE,
                                name='player')
        input_vars = [walls, cards, player]

        flattened = [tf.contrib.layers.flatten(t)
                     for t in (walls, cards, player)]
        combined_input = tf.concat(1, flattened, name='combined_input')

        fc = tf.contrib.layers.fully_connected
        if self.options.bias_only:
            predict_op = fc(0.0 * combined_input, trainable=True,
                            activation_fn=tf.identity, num_outputs=len(cards_env.ACTIONS))
            with tf.variable_scope('fully_connected', reuse=True):
                biases = tf.get_variable('biases')
        else:
            hidden1 = fc(combined_input, trainable=True,
                         num_outputs=self.options.pg_hidden_size)
            predict_op = fc(hidden1, trainable=True, activation_fn=tf.identity,
                            num_outputs=len(cards_env.ACTIONS))
            with tf.variable_scope('fully_connected_1', reuse=True):
                biases = tf.get_variable('biases')
            print_node = tf.Print(predict_op, [biases], message='biases: ', summarize=10)
            with tf.control_dependencies([print_node]):
                predict_op = tf.identity(predict_op)

        return input_vars, predict_op

    @profile
    def train_one_batch(self, insts, env, t):
        env.configure([inst.input for inst in insts], verbosity=self.options.verbosity)
        observation = env._get_obs()
        info = None
        self.init_belief(env, observation)

        if self.options.verbosity >= 1:
            progress.start_task('Step', self.options.max_steps)

        for step in range(self.options.max_steps):
            if self.options.verbosity >= 1:
                progress.progress(step)

            if self.options.render:
                env.render()
            actions = self.action(env, observation, info, testing=False)
            prev_obs = observation
            observation, reward, done, info = env.step(actions)
            self.update_belief(env, prev_obs, actions, observation, reward, done, info)
            if all(done):
                break

        '''
        from tensorflow.python.client import timeline
        trace = timeline.Timeline(step_stats=self.run_metadata.step_stats)

        with config.open('timeline.ctf.json', 'w') as trace_file:
            trace_file.write(trace.generate_chrome_trace_format())
        '''

        rewards = np.array(self.rewards)  # max_steps x batch_size
        done = np.array(self.done, dtype=np.int32)  # max_steps x batch_size
        actions = np.array(self.actions).reshape(rewards.shape)
        # force actions on steps where reward is zero (already done) to nop
        actions[1:, :] *= (1 - done)[:-1, :]
        for game in range(rewards.shape[1]):
            action_hist = np.bincount(actions[:, game],
                                      minlength=len(cards_env.ACTIONS)).tolist()
            if self.options.verbosity >= 7:
                print('Total reward: {}  {}'.format(rewards[:, game].sum(), action_hist))
        total_rewards = np.repeat(rewards.sum(axis=0), rewards.shape[0])
        assert total_rewards.shape == (rewards.shape[0] * rewards.shape[1],), \
            (total_rewards.shape, rewards.shape)
        credit = np.ones(done.shape)
        credit[1:, :] *= 1.0 - done[:-1, :]
        credit = credit.ravel()  # (credit / credit.sum(axis=0)).ravel()
        assert credit.shape == total_rewards.shape, (credit.shape, total_rewards.shape)

        if self.options.verbosity >= 1:
            progress.end_task()

        feed_dict = self.batch_inputs(self.inputs[:-cards_env.MAX_BATCH_SIZE])
        for label, value in zip(self.label_vars, [np.array(self.actions),
                                                  total_rewards,
                                                  credit]):
            feed_dict[label] = value
        self.run_train(feed_dict)

    def action(self, env, observations, info, testing=True):
        inputs = self.preprocess(observations, env)
        feed_dict = self.batch_inputs(inputs)
        dist = self.run_predict(feed_dict=feed_dict)
        #                       options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        #                       run_metadata=self.run_metadata)
        random_epsilon = 0.0 if testing else self.options.pg_random_epsilon
        if self.options.move_only:
            dist[:, 0] = -float('inf')
            dist[:, 5:] = -float('inf')
        actions = sample(dist, random_epsilon=random_epsilon)
        self.actions.extend(actions)
        return actions

    def preprocess(self, observations, env):
        processed = []
        for start in range(0, len(observations), 6):
            walls, cards, player, hand, floor, language = observations[start:start + 6]
            processed.append([walls, cards, player])
        return processed

    def batch_inputs(self, inputs):
        feed_dict = {}
        for i, input_var in enumerate(self.input_vars):
            feed_dict[input_var] = np.array([inp[i] for inp in inputs])
        return feed_dict

    def init_belief(self, env, observations):
        self.actions = []
        self.rewards = []
        self.done = []
        self.inputs = self.preprocess(observations, env)

    def update_belief(self, env, prev_obs, actions, observations, rewards, done, info):
        self.rewards.append(rewards)
        self.done.append(done)
        self.inputs.extend(self.preprocess(observations, env))


class RLListenerLearner(KarpathyPGLearner):
    def train(self, training_instances, validation_instances='ignored', metrics='ignored'):
        self.init_vectorizers()
        super(RLListenerLearner, self).train(training_instances, validation_instances, metrics)

    def init_vectorizers(self):
        if self.options.verbosity >= 4:
            print('Initializing vectorizer')
        unk_threshold = self.options.unk_threshold
        self.seq_vec = vectorizers.SequenceVectorizer(unk_threshold=unk_threshold)
        for trans in datasets.train_transcripts():
            for event in trans.events:
                if event.action == cards.UTTERANCE:
                    utt = ['<s>'] + event.parse_contents() + ['</s>']
                    self.seq_vec.add(utt)

    def get_predict_op(self):
        # Inputs
        walls = tf.placeholder(tf.float32, shape=(None,) + cards_env.MAX_BOARD_SIZE,
                               name='walls')
        cards = tf.placeholder(tf.float32, shape=(None,) + cards_env.MAX_BOARD_SIZE,
                               name='cards')
        player = tf.placeholder(tf.float32, shape=(None,) + cards_env.MAX_BOARD_SIZE,
                                name='player')
        utt = tf.placeholder(tf.int32, shape=(None, self.seq_vec.max_len),
                             name='utt')
        utt_len = tf.placeholder(tf.int32, shape=(None,), name='utt_len')
        input_vars = [walls, cards, player, utt, utt_len]

        # Hidden layers
        cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.options.num_rnn_units,
                                            use_peephole=False)

        embeddings = tf.Variable(tf.random_uniform([self.seq_vec.num_types,
                                                    self.options.embedding_size], -1.0, 1.0),
                                 name='embeddings')

        utt_embed = tf.nn.embedding_lookup(embeddings, utt, name='utt_embed')
        utt_embed_drop = tf.nn.dropout(utt_embed, keep_prob=self.dropout_keep_prob,
                                       name='utt_embed_drop')
        _, encoder_state_tuple = tf.nn.dynamic_rnn(cell, utt_embed_drop, sequence_length=utt_len,
                                                   dtype=tf.float32)
        encoder_state = tf.concat(1, encoder_state_tuple, name='encoder_state')
        encoder_state_drop = tf.nn.dropout(encoder_state, keep_prob=self.dropout_keep_prob,
                                           name='encoder_state_drop')

        flattened = [tf.contrib.layers.flatten(t)
                     for t in (walls, cards, player, encoder_state_drop)]
        combined_input = tf.concat(1, flattened, name='combined_input')

        fc = tf.contrib.layers.fully_connected
        if self.options.bias_only:
            predict_op = fc(0.0 * combined_input, trainable=True,
                            activation_fn=tf.identity, num_outputs=len(cards_env.ACTIONS))
            with tf.variable_scope('fully_connected', reuse=True):
                biases = tf.get_variable('biases')
        else:
            hidden1 = fc(combined_input, trainable=True,
                         num_outputs=self.options.pg_hidden_size)
            predict_op = fc(hidden1, trainable=True, activation_fn=tf.identity,
                            num_outputs=len(cards_env.ACTIONS))
            with tf.variable_scope('fully_connected_1', reuse=True):
                biases = tf.get_variable('biases')
                biases_down = tf.slice(biases, [4], [1], name='biases_down')
                predict_down = tf.slice(predict_op, [0, 4], [-1, 1], name='predict_down')
            print_nodes = [
                tf.Print(predict_op, [biases_down], message='biases_down: ', summarize=10),
                tf.Print(predict_op, [predict_down], message='predict_down: ', summarize=10)
            ]
            with tf.control_dependencies(print_nodes):
                predict_op = tf.identity(predict_op)

        return input_vars, predict_op

    def preprocess(self, observations, env):
        processed = []
        for w in range(0, len(observations) / 6):
            walls, cards, player, hand, floor, language = observations[w * 6:(w + 1) * 6]
            if language[0]:
                utt_tokens = env.get_language_obs(w, 0)[0]
            else:
                utt_tokens = ['<s>', '</s>']
            utt = self.seq_vec.vectorize(utt_tokens)
            utt_len = len(utt_tokens)
            processed.append([walls, cards, player, utt, utt_len])
        return processed


def sample(a, temperature=1.0, random_epsilon=0.0, verbose=False):
    # helper function to sample an index from a log probability array
    a = np.array(a)
    if len(a.shape) < 1:
        raise ValueError('scalar is not a valid probability distribution')
    elif len(a.shape) == 1:
        # Cast to higher resolution to try to get high-precision normalization
        a = np.exp((a / temperature).astype(np.float64))
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
