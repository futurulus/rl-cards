import argparse
import gzip
import numpy as np
import tensorflow as tf

from stanza.monitoring import progress
from stanza.research import config, iterators
from stanza.research.learner import Learner

import cards_env
import tfutils
import world
import vectorizers
from rl_learner import TensorflowLearner


parser = config.get_options_parser()
parser.add_argument('--train_epochs', type=int, default=10,
                    help='The number of passes through the data to make in training '
                         'reflex agents.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='The size of reflex agent batches for training.')
parser.add_argument('--eval_batch_size', type=int, default=128,
                    help='The size of reflex agent batches for testing.')
parser.add_argument('--unk_threshold', type=int, default=1,
                    help='Maximum number of occurrences for a token to occur in training '
                         'and still be counted as an unknown word (<unk>).')

parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='The learning rate for neural sequence model training.')
parser.add_argument('--embedding_size', type=int, default=128,
                    help='The dimensionality of word embeddings in neural sequence models.')
parser.add_argument('--num_rnn_units', type=int, default=128,
                    help='The dimensionality of the RNN state in neural sequence models.')


NUM_LOCS = np.prod(cards_env.MAX_BOARD_SIZE)


def trace_nans(op, session, feed_dict, indent=0, seen=None):
    if seen is None:
        seen = set()

    if id(op) in seen:
        return

    seen.add(id(op))

    try:
        inputs = list(op.inputs)
    except NameError:
        inputs = []

    if not inputs:
        return

    for inp in inputs:
        try:
            val = session.run(inp, feed_dict=feed_dict)
            tag = ''
        except Exception:
            val = np.array([-float('inf')])
            tag = ' <not fetchable>'

        try:
            good = np.isfinite(val).all()
            if good:
                tag += ' <good>'
            else:
                tag += ' <bad!>'
        except TypeError:
            good = None
            tag += ' <finite? {}>'.format(val.dtype)

        if val is not None and good is True:
            if 'fetch' not in tag:
                print(' ' * indent + inp.name + tag)
        else:
            child_op = inp._op
            if 'fetch' in tag:
                new_indent = indent
            else:
                print(' ' * indent + inp.name + tag)
                new_indent = indent + 1
            trace_nans(child_op, session, feed_dict, indent=new_indent, seen=seen)


TRAIN_EVENTS = 272879
ACTION_DIST = {
    world.cards.MOVE: 224163. / TRAIN_EVENTS,
    world.cards.UTTERANCE: 27455. / TRAIN_EVENTS,
    world.cards.PICKUP: 11316. / TRAIN_EVENTS,
    world.cards.DROP: 7183. / TRAIN_EVENTS,
}


class UniformListener(Learner):
    def __init__(self):
        self.get_options()

    def get_options(self):
        if not hasattr(self, 'options'):
            options = config.options()
            self.options = argparse.Namespace(**options.__dict__)
        return self.options

    def train(self, training_instances, validation_instances, metrics):
        pass

    @property
    def num_params(self):
        return 0

    def predict_and_score(self, eval_instances, random=False, verbosity=0):
        predictions = []
        scores = []

        if self.options.verbosity + verbosity >= 1:
            progress.start_task('Instance', len(eval_instances))

        all_cards = [r + s for r in cards_env.RANKS for s in cards_env.SUITS]
        cards_to_loc = {k: (1, 1) for k in all_cards}

        for i, inst in enumerate(eval_instances):
            if self.options.verbosity + verbosity >= 1:
                progress.progress(i)

            walls = inst.input['walls']
            num_possible_locs = np.ones(walls.shape).sum() - walls.sum()
            predictions.append(world.build_world(walls, dict(cards_to_loc)).__dict__)
            score = -len(all_cards) * np.log(num_possible_locs + 3.0) - np.log(num_possible_locs)
            scores.append(score)

        if self.options.verbosity + verbosity >= 1:
            progress.end_task()

        return predictions, scores

    def dump(self, prefix):
        '''
        :param prefix: The *path prefix* (a string, not a file-like object!)
                       of the model file to be written ('.pkl' will be added)
        '''
        with open(prefix + '.pkl', 'wb') as outfile:
            super(UniformListener, self).dump(outfile)

    def load(self, filename):
        '''
        :param filename: The *path* (a string, not a file-like object!)
                         of the model file to be read
        '''
        with open(filename, 'rb') as infile:
            super(UniformListener, self).load(infile)


class ReflexListener(TensorflowLearner):
    def __init__(self):
        self.get_options()

    def get_options(self):
        if not hasattr(self, 'options'):
            options = config.options()
            self.options = argparse.Namespace(**options.__dict__)
        return self.options

    def train(self, training_instances, validation_instances, metrics):
        self.init_vectorizers(training_instances)

        self.build_graph()
        self.init_params()

        batches = iterators.gen_batches(training_instances,
                                        batch_size=self.options.batch_size)

        if self.options.verbosity >= 1:
            progress.start_task('Epoch', self.options.train_epochs)

        for epoch in range(self.options.train_epochs):
            if self.options.verbosity >= 1:
                progress.progress(epoch)

            if self.options.verbosity >= 1:
                progress.start_task('Batch', len(batches))

            for i, batch in enumerate(batches):
                if self.options.verbosity >= 1:
                    progress.progress(i)

                batch = list(batch)
                feed_dict = self.vectorize_inputs(batch)
                feed_dict.update(self.vectorize_labels(batch))
                self.run_train(feed_dict)

            if self.options.verbosity >= 1:
                progress.end_task()

        if self.options.verbosity >= 1:
            progress.end_task()

    def predict_and_score(self, eval_instances, random=False, verbosity=0):
        predictions = []
        scores = []

        batches = iterators.gen_batches(eval_instances,
                                        batch_size=self.options.eval_batch_size)

        with gzip.open(config.get_file_path('dists.b64.gz'), 'w'):
            pass

        if self.options.verbosity + verbosity >= 1:
            progress.start_task('Eval batch', len(batches))

        for i, batch in enumerate(batches):
            if self.options.verbosity + verbosity >= 1:
                progress.progress(i)

            batch = list(batch)

            inputs = self.vectorize_inputs(batch)
            output = self.session.run(self.predict_op, feed_dict=inputs)
            predictions_batch = self.output_to_preds(output, batch)
            predictions.extend(predictions_batch)
            labels = self.vectorize_labels(batch)
            scores_batch = self.output_to_scores(output, labels)
            scores.extend(scores_batch)

        if self.options.verbosity + verbosity >= 1:
            progress.end_task()

        return predictions, scores

    def init_vectorizers(self, training_instances):
        unk_threshold = self.options.unk_threshold
        self.seq_vec = vectorizers.SequenceVectorizer(unk_threshold=unk_threshold)
        self.seq_vec.add_all(inst.input['utt'] for inst in training_instances)

    def get_layers(self):
        # Inputs
        walls = tf.placeholder(tf.float32, shape=(None,) + cards_env.MAX_BOARD_SIZE,
                               name='walls')
        cards = tf.placeholder(tf.float32, shape=(None,) + cards_env.MAX_BOARD_SIZE,
                               name='cards')
        utt = tf.placeholder(tf.int32, shape=(None, self.seq_vec.max_len),
                             name='utt')
        utt_len = tf.placeholder(tf.int32, shape=(None,), name='utt_len')
        input_vars = [walls, cards, utt, utt_len]

        # Hidden layers
        cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.options.num_rnn_units,
                                            use_peephole=False)

        embeddings = tf.Variable(tf.random_uniform([self.seq_vec.num_types,
                                                    self.options.embedding_size], -1.0, 1.0),
                                 name='embeddings')

        utt_embed = tf.nn.embedding_lookup(embeddings, utt)
        '''
        # Gives an error:
        # Dimension must be 2 but is 3 for 'transpose' (op: 'Transpose') with input shapes:
        #   [?,24], [3].
        embed_wrap = tf.nn.rnn_cell.EmbeddingWrapper(
            cell, embedding_classes=self.seq_vec.num_types,
            embedding_size=self.options.embedding_size)
        '''
        _, encoder_state_tuple = tf.nn.dynamic_rnn(cell, utt_embed, sequence_length=utt_len,
                                                   dtype=tf.float32)
        encoder_state = tf.concat(1, encoder_state_tuple)

        full_linear = self.state_to_linear_dist(encoder_state)

        walls_mask = tf.reshape(tf.where(walls <= 0.5,
                                         tf.zeros_like(walls),
                                         -44.0 * tf.ones_like(walls),
                                         name='walls_mask_2d'),
                                [-1, NUM_LOCS], name='walls_mask')

        p2_loc_linear = tf.slice(full_linear, [0, 0], [-1, NUM_LOCS], name='p2_loc_linear')
        p2_loc_masked = tf.add(p2_loc_linear, walls_mask, name='p2_loc_masked')
        p2_loc_dist = tf.nn.log_softmax(p2_loc_masked, name='p2_loc_dist')

        cards_mask = tf.reshape(tf.where(tf.abs(cards) > 0.5,
                                         tf.zeros_like(cards),
                                         -44.0 * tf.ones_like(cards),
                                         name='cards_mask_2d'),
                                [-1, 1, NUM_LOCS], name='cards_mask')
        all_mask_board = tf.add(cards_mask, tf.expand_dims(walls_mask, 1), name='all_mask')
        mask_special_cols = tf.zeros_like(tf.slice(cards_mask, [0, 0, 0], [-1, -1, 2]),
                                          name='mask_special_cols')
        all_mask = tf.concat(2, [mask_special_cols, all_mask_board], name='all_mask')

        card_loc_linear = tf.slice(full_linear, [0, NUM_LOCS], [-1, -1], name='card_loc_linear')
        card_loc_rows = tf.reshape(card_loc_linear, [-1, 52, NUM_LOCS + 2], name='card_loc_rows')
        card_loc_masked = tf.add(card_loc_rows, all_mask, name='card_loc_masked')
        card_loc_dist = tf.nn.log_softmax(card_loc_masked, name='card_loc_dist')

        predict_op = (card_loc_dist, p2_loc_dist)

        # Labels
        true_card_loc = tf.placeholder(tf.int32, shape=(None, 52), name='true_card_loc')
        true_p2_loc = tf.placeholder(tf.int32, shape=(None,), name='true_p2_loc')
        label_vars = [true_card_loc, true_p2_loc]

        # Loss function
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits
        card_loc_loss = xent(card_loc_masked, true_card_loc, name='card_loc_loss')
        p2_loc_loss = xent(p2_loc_masked, true_p2_loc, name='p2_loc_loss')
        loss = tf.reduce_mean(tf.reduce_sum(card_loc_loss, 1, name='card_loc_loss_sum') +
                              p2_loc_loss, name='loss')

        # Optimizer
        opt = tf.train.RMSPropOptimizer(learning_rate=self.options.learning_rate)
        var_list = tf.trainable_variables()
        if self.options.verbosity >= 4:
            print('Trainable variables:')
            for var in var_list:
                print(var.name)
        train_op = tfutils.minimize_with_grad_clip(opt, self.options.pg_grad_clip,
                                                   loss, var_list=var_list)

        return input_vars, label_vars, train_op, predict_op

    def state_to_linear_dist(self, encoder_state):
        fc = tf.contrib.layers.fully_connected
        with tf.variable_scope('hidden'):
            hidden = fc(encoder_state, trainable=True, activation_fn=tf.identity,
                        num_outputs=NUM_LOCS + 52 * (NUM_LOCS + 2))
        return hidden

    def vectorize_inputs(self, batch):
        walls = np.array([inst.input['walls'] for inst in batch])
        cards = np.array([inst.input['cards'] for inst in batch])
        utt = self.seq_vec.vectorize_all([inst.input['utt'] for inst in batch])
        utt_len = np.array([len(inst.input['utt']) for inst in batch])
        return dict(zip(self.input_vars, [walls, cards, utt, utt_len]))

    def vectorize_labels(self, batch):
        card_names = [rank + suit for rank in cards_env.RANKS for suit in cards_env.SUITS]
        true_card_loc = np.array([
            [coord_to_loc_index(inst.output['cards_to_loc'][card], card=True)
             for card in card_names]
            for inst in batch
        ])
        true_p2_loc = np.array([
            coord_to_loc_index(inst.output['p2_loc'])
            for inst in batch
        ])
        return dict(zip(self.label_vars, [true_card_loc, true_p2_loc]))

    def output_to_preds(self, output, batch):
        card_names = [rank + suit for rank in cards_env.RANKS for suit in cards_env.SUITS]
        num_cards = len(card_names)
        card_loc_rows, p2_loc = output

        assert card_loc_rows.shape[1:] == (num_cards, NUM_LOCS + 2), card_loc_rows.shape
        assert p2_loc.shape[1:] == (NUM_LOCS,), card_loc_rows.shape

        with gzip.open(config.get_file_path('dists.b64.gz'), 'a') as outfile:
            for row in summarize_output(card_loc_rows, p2_loc):
                outfile.write(row)
                outfile.write('\n')

        card_loc_indices = card_loc_rows.argmax(axis=2)
        p2_loc_indices = p2_loc.argmax(axis=1)

        preds = []
        for i, inst in enumerate(batch):
            cards_to_loc_pred = {
                name: loc_index_to_coord(idx, card=True)
                for name, idx in zip(card_names, card_loc_indices[i])
            }
            p2_loc_pred = loc_index_to_coord(p2_loc_indices[i])
            state = world.build_world(inst.input['walls'], cards_to_loc_pred, p2_loc=p2_loc_pred)
            preds.append(state.__dict__)
        return preds

    def output_to_scores(self, output, labels):
        card_loc_rows, p2_loc = output
        true_card_loc, true_p2_loc = [labels[t] for t in self.label_vars]
        card_loc_scores = card_loc_rows[np.arange(true_card_loc.shape[0])[:, np.newaxis],
                                        np.arange(true_card_loc.shape[1]), true_card_loc]
        assert card_loc_scores.shape == (card_loc_rows.shape[0], card_loc_rows.shape[1]), \
            card_loc_scores.shape
        p2_loc_scores = p2_loc[np.arange(card_loc_rows.shape[0]), true_p2_loc]
        assert p2_loc_scores.shape == (p2_loc.shape[0],), \
            p2_loc_scores.shape
        result = card_loc_scores.sum(axis=1) + p2_loc_scores
        return [float(s) for s in result]


class FactoredReflexListener(ReflexListener):
    def state_to_linear_dist(self, encoder_state):
        fc = tf.contrib.layers.fully_connected
        with tf.variable_scope('referent'):
            ref = fc(encoder_state, trainable=True, activation_fn=tf.identity,
                     num_outputs=53)
        with tf.variable_scope('loc'):
            loc = fc(encoder_state, trainable=True, activation_fn=tf.identity,
                     num_outputs=NUM_LOCS + 2)

        # log(x + y) = log((x/y + 1)*y) = softplus(log(x/y)) + log(y)
        #                               = softplus(log(x) - log(y)) + log(y)
        # log(sigmoid(x)) = log(1/(1 + exp(-x))) = -log(1 + exp(-x)) = -softplus(-x)
        # softplus(x) - softplus(-x) = log(exp(x) + 1) - log(exp(-x) + 1)
        #                            = log(exp(x) + 1) - (-x) - log(1 + exp(x)) = x
        #
        #     Let x = sigmoid(ref) * softmax(loc), y = sigmoid(-ref) * 1/(NUM_LOCS+2).
        #     => log(x) = log(sigmoid(ref) * softmax(loc)) = -softplus(-ref) + log_softmax(loc),
        #        log(y) = log(sigmoid(-ref) / (NUM_LOCS+2)) = -softplus(ref) - log(NUM_LOCS+2)
        #
        # log(sigmoid(ref) * softmax(loc) + sigmoid(-ref) * 1/(NUM_LOCS+2)) =
        #      softplus(log_softmax(loc) + softplus(ref) - softplus(-ref) + log(NUM_LOCS+2)) -
        #              softplus(ref) - log(NUM_LOCS+2) =
        #      softplus(log_softmax(loc) + ref + log(NUM_LOCS+2)) - softplus(ref) - log(NUM_LOCS+2)
        ref_expanded = tf.expand_dims(ref, 2)
        loc_expanded = tf.expand_dims(loc, 1)
        log_maxent = tf.mul(np.log(1. / (NUM_LOCS + 2), dtype=np.float32),
                            tf.ones_like(loc_expanded),
                            name='log_maxent')
        big_softplus = tf.nn.softplus(tf.nn.log_softmax(loc_expanded) + ref_expanded - log_maxent,
                                      name='big_softplus')
        small_softplus = tf.nn.softplus(ref_expanded, name='small_softplus')
        product = tf.sub(big_softplus, small_softplus - log_maxent, name='product')

        p2_row = tf.squeeze(tf.slice(product, [0, 0, 0], [-1, 1, NUM_LOCS]), 1, name='p2_row')
        cards_row = tf.reshape(tf.slice(product, [0, 1, 0], [-1, -1, -1]),
                               [-1, 52 * (NUM_LOCS + 2)], name='cards_row')

        return tf.concat(1, [p2_row, cards_row], name='linear_dist_concat')


def loc_index_to_coord(idx, card=False):
    '''
    >>> loc_index_to_coord(37, card=False)  # 37 = 1 * 34 + 3
    (1, 3)
    >>> loc_index_to_coord(37, card=True)
    (1, 1)
    >>> loc_index_to_coord(1, card=True)
    'Player 2'
    >>> loc_index_to_coord(0, card=True)
    '''
    if card:
        if idx == 0:
            return None
        elif idx == 1:
            return world.cards.PLAYER2
        else:
            idx -= 2

    return np.unravel_index(idx, cards_env.MAX_BOARD_SIZE)


def coord_to_loc_index(coord, card=False):
    '''
    >>> coord_to_loc_index((1, 1), card=False)  # 35 = 1 * 34 + 1
    35
    >>> coord_to_loc_index((1, 1), card=True)
    37
    >>> coord_to_loc_index('Player 2', card=True)
    1
    >>> coord_to_loc_index(None, card=True)
    0
    '''
    if card:
        try:
            length = len(coord)
        except TypeError:
            length = None

        if coord == world.cards.PLAYER2:
            return 1
        elif length != 2:
            return 0

    idx = np.ravel_multi_index(coord, cards_env.MAX_BOARD_SIZE)
    if card:
        return idx + 2
    else:
        return idx


def summarize_output(card_loc_rows, p2_loc):
    '''
    >>> summarize_output([
    ...     [[-7.0, -8.0], [-50.0, -7.0]],
    ...     [[-8.0, -7.0], [-50.0, -7.0]],
    ... ], [[-8.0, -7.0], [-7.0, -8.0]])
    ['v7F', 'F7v']
    '''
    # 71 07 17  17 07 71
    # 57  7 15  15  7 57
    #  v  7  F   F  7  v
    rows = []
    card_loc_rows = np.array(card_loc_rows)
    p2_loc = np.array(p2_loc)

    for i in range(card_loc_rows.shape[0]):
        row = np.concatenate([card_loc_rows[i].ravel(), p2_loc[i]])
        assert row.shape == (card_loc_rows.shape[1] * card_loc_rows.shape[2] + p2_loc.shape[1],), \
            row.shape
        row_quant = quantize(row)
        rows.append(''.join(base64_char(row_quant[i], row_quant[i + 1])
                            for i in range(0, row_quant.shape[0], 2)))
    return rows


def quantize(row):
    '''
    >>> quantize([-48.0, -7.0, -8.0, -9.0])
    array([0, 7, 4, 1])
    '''
    row = np.copy(row)
    row[row <= -30.0] = float('nan')
    lower, upper = np.nanmin(row), np.nanmax(row)
    result = np.zeros(row.shape, dtype=np.int)
    offsets = (row[np.isfinite(row)] - lower) / (upper - lower)
    result[np.isfinite(row)] = np.minimum((offsets * 7.0 + 1.0).astype(np.int), 7)
    return result


def base64_char(n1, n2):
    '''
    0 through 9, A through Z, a through z, '.', '/'

    >>> base64_char(7, 1)
    'v'
    >>> base64_char(1, 7)
    'F'
    >>> base64_char(0, 7)
    '7'
    '''
    assert 0 <= n1 < 8 and 0 <= n2 < 8, (n1, n2)
    combined = n1 * 8 + n2
    if combined < 10:
        return str(combined)
    elif combined < 36:
        return chr(ord('A') + (combined - 10))
    elif combined < 62:
        return chr(ord('a') + (combined - 36))
    else:
        return chr(ord('.') + (combined - 62))
