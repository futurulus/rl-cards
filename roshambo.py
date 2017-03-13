import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import logsumexp
import tensorflow as tf
from multiprocessing import Pool

import tfutils

BATCH_SIZE = 10
NUM_BATCHES = 500
NUM_ITERS = 5
NUM_TEST_GAMES = 1000


class Agent(object):
    def __init__(self, num_options):
        self.num_options = num_options

    def train(self):
        for b in range(NUM_BATCHES):
            perfect = self.train_one_batch()
            if perfect:
                return b
        return NUM_BATCHES

    def train_one_batch(self):
        pairs = []
        perfect = True
        for _ in range(BATCH_SIZE):
            games = []
            while True:
                opponent, our_move, win = self.play_game(explore=True)
                # sys.stdout.write('{}{} '.format(opponent, our_move))
                # sys.stdout.flush()
                games.append((opponent, our_move))
                if win:
                    break
            reward = 1.0 / len(games)
            if len(games) > 1:
                perfect = False
            # print(reward)
            pairs.append((games, reward))
        self.update(pairs)
        return perfect

    def test(self):
        num_wins = 0
        for _ in range(NUM_TEST_GAMES):
            _, _, win = self.play_game()
            if win:
                num_wins += 1
        return 1.0 * num_wins / NUM_TEST_GAMES

    def play_game(self, explore=False):
        opponent = np.random.randint(self.num_options)
        winning_move = (opponent + 1) % self.num_options
        our_move = self.action(opponent, explore=explore)
        return opponent, our_move, (our_move == winning_move)

    def update(self, pairs):
        raise NotImplementedError

    def action(self, explore):
        raise NotImplementedError


class RandomAgent(Agent):
    def update(self, pairs):
        pass

    def action(self, opponent, explore):
        return np.random.randint(self.num_options)


class RLAgent(Agent):
    def __init__(self, num_options):
        super(RLAgent, self).__init__(num_options)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.mask = tf.placeholder(tf.float32, shape=(None, None))
            self.opponent = tf.placeholder(tf.int32, shape=(None, None))
            self.our_move = tf.placeholder(tf.int32, shape=(None, None))
            self.reward = tf.placeholder(tf.float32, shape=(None, None))

            self.policy = tf.Variable(tf.zeros([num_options, num_options]))
            self.action_dist = tf.gather(self.policy, self.opponent)

            logp = tf.nn.sparse_softmax_cross_entropy_with_logits(self.action_dist, self.our_move)
            signal = -logp * self.reward * self.mask

            loss = tf.reduce_mean(-signal)

            opt = tf.train.RMSPropOptimizer(learning_rate=0.1)
            self.train_op = tfutils.minimize_with_grad_clip(opt, 5.0, loss,
                                                            var_list=[self.policy])
        self.session = tf.Session(graph=self.graph)

        with self.graph.as_default():
            tf.global_variables_initializer().run(session=self.session)

        self.policies = []

    def update(self, pairs):
        feed_dict = self.get_feed_dict_update(pairs)
        self.session.run(self.train_op, feed_dict=feed_dict)
        policy = self.session.run(self.policy)
        self.policies.append(policy - logsumexp(policy, axis=1, keepdims=True))

    def action(self, opponent, explore):
        feed_dict = self.get_feed_dict_action(opponent)
        logdist = self.session.run(self.action_dist, feed_dict=feed_dict)[0][0]
        dist = np.exp(logdist - logsumexp(logdist))
        if explore:
            dist = dist.astype(np.float64)
            dist /= np.sum(dist, dtype=np.float64)
            return np.argmax(np.random.multinomial(1, dist, 1)[0])
        else:
            return np.argmax(dist)

    def get_feed_dict_update(self, pairs):
        games, rewards = zip(*pairs)
        max_rounds = max(len(g) for g in games)
        mask = np.array([
            [1.0] * len(g) + [0.0] * (max_rounds - len(g))
            for g in games
        ])
        credit = mask * (np.array(rewards) - 1/3.)[:, np.newaxis]
        games_arr = np.array([
            g + [(0, 0)] * (max_rounds - len(g))
            for g in games
        ])
        opponent = games_arr[:, :, 0]
        our_move = games_arr[:, :, 1]
        return {
            self.mask: mask,
            self.opponent: opponent,
            self.our_move: our_move,
            self.reward: credit,
        }

    def get_feed_dict_action(self, opponent):
        return {self.opponent: np.array([[opponent]])}


def train_o(o):
    try:
        print(o)
        return RLAgent(o).train()
    except KeyboardInterrupt:
        return NUM_BATCHES


def rl_test():
    ovals = range(2, 8) + range(8, 32, 2)
    pool = Pool(16)
    num_batches = np.array(pool.map(train_o, ovals * NUM_ITERS)).reshape((NUM_ITERS, len(ovals)))
    medians = np.median(num_batches, axis=0)
    plt.plot(ovals, medians)
    plt.show()


if __name__ == '__main__':
    rl_test()
