from stanza.research import config, instance


parser = config.get_options_parser()
parser.add_argument('--game_config', default='Ace1P',
                    help='The type of game configuration (defines rewards and speaker models '
                         'for the Cards environment).')
parser.add_argument('--p2_learner', default=None,
                    help='The class of the pre-trained model to use as Player 2 in two-player '
                         'training configs.')
parser.add_argument('--p2_load', default=None,
                    help='The prefix of the pre-trained model to use as Player 2 in two-player '
                         'training configs.')


def new(key):
    '''
    Construct a new config with the class named by `key`. A list
    of available configs is in the dictionary `CONFIGS`.
    '''
    return CONFIGS[key]()


class Config(object):
    @property
    def num_players(self):
        return 2

    def get_all_language_obs(self, env, w):
        return [None for _ in range(len(self.num_players))]

    def value(self, env, w):
        return 0.0

    def action_reward(self, env, w, action):
        return -1.0


class Ace1P(Config):
    @property
    def num_players(self):
        return 1


class AceModelS(Config):
    def __init__(self):
        import learners
        import cards_env
        options = config.options()
        if options.verbosity >= 4:
            print('Loading speaker')
        self.speaker = learners.new(options.p2_learner)
        self.speaker.load(options.p2_load)
        self.utterances = [None for _ in range(cards_env.MAX_BATCH_SIZE)]
        self.ace_locs = [None for _ in range(cards_env.MAX_BATCH_SIZE)]

    def update_language_obs(self, env):
        update_ws = []
        insts = []
        for w in range(len(self.ace_locs)):
            ace_loc = env.cards_to_loc[w][(0, 0)]
            if ace_loc != self.ace_locs[w]:
                insts.append(self.build_inst(env, w))
                update_ws.append(w)
                self.ace_locs[w] = ace_loc
        if insts:
            new_utts = self.speaker.predict(insts, random=True)
            for w, utt in zip(update_ws, new_utts):
                self.utterances[w] = utt

    def get_language_obs(self, env, w):
        return [None, self.utterances[w]]

    def build_inst(self, env, w):
        return instance.Instance(input={'loc': env.cards_to_loc[w][(0, 0)],
                                        'walls': env.walls[w]},
                                 output=['<s>', '</s>'])

    def value(self, env, w):
        px, py = env.p1_loc[w]
        cx, cy = env.cards_to_loc[w][(0, 0)]
        # Manhattan distance
        return abs(px - cx) + abs(py - cy)

    def action_reward(self, env, w, action):
        return 0.0


CONFIGS = {
    'Ace1P': Ace1P,
    'AceModelS': AceModelS,
}
