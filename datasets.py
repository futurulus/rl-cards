import csv
from collections import namedtuple
import numpy as np

from stanza.research.rng import config
from stanza.research.instance import Instance
from stanza.research.rng import get_rng

from world import CardsWorld, build_world, MAX_BOARD_SIZE, cards
import world
from cards_cache import all_transcripts


rng = get_rng()

parser = config.get_options_parser()
parser.add_argument('--line_of_sight', type=int, default=3,
                    help='Maximum distance (Manhattan distance) from the listener that a card is '
                         'visible. If negative, line of sight is infinite.')
parser.add_argument('--line_of_sight_p2', type=int, default=-1,
                    help='Maximum distance (Manhattan distance) from the speaker that a card is '
                         'visible. If negative, line of sight is infinite.')
parser.add_argument('--dist_num_rows', type=int, default=2,
                    help='Board height. Used only in "dist" data_source.')
parser.add_argument('--dist_num_cols', type=int, default=1,
                    help='Board height. Used only in "dist" data_source.')
parser.add_argument('--dist_offset_row', type=int, default=1,
                    help='Which row to place the ace of spades in '
                         '(relative to player position). '
                         'Used only in "dist" data_source.')
parser.add_argument('--dist_offset_col', type=int, default=0,
                    help='Which column to place the ace of spades in '
                         '(relative to player position). '
                         'Used only in "dist" data_source.')
parser.add_argument('--sampler_model_learner', default=None,
                    help='The class of the model to draw samples from in co-training. Used '
                         'only in "samples_*_to_*" data_sources.')
parser.add_argument('--sampler_model_load', default=None,
                    help='The prefix of the model to draw samples from in co-training. Used '
                         'only in "samples_*_to_*" data_sources.')
parser.add_argument('--num_samples', type=int, default=10000,
                    help='Number of samples to draw in co-training. Used '
                         'only in "samples_*_to_*" data_sources.')


def train_transcripts():
    return [trans for n, trans in enumerate(all_transcripts())
            if n % 10 < 6]


def dev_transcripts():
    return [trans for n, trans in enumerate(all_transcripts())
            if n % 10 in (6, 7)]


def test_transcripts():
    return [trans for n, trans in enumerate(all_transcripts())
            if n % 10 in (8, 9)]


def cards_train():
    insts = [Instance(input=CardsWorld(trans), output=0)
             for trans in train_transcripts()]
    rng.shuffle(insts)
    return insts


def cards_dev():
    return [Instance(input=CardsWorld(trans), output=0)
            for trans in dev_transcripts()]


def cards_test():
    return [Instance(input=CardsWorld(trans), output=0)
            for trans in test_transcripts()]


def single_loc_train():
    insts = cards_train()
    for inst in insts:
        inst.input.cards_to_loc = {'AS': (1, 10)}
    return insts


def single_loc_dev():
    insts = cards_dev()
    for inst in insts:
        inst.input.cards_to_loc = {'AS': (1, 10)}
    return insts


def only_ace_train():
    insts = cards_train()
    for inst in insts:
        inst.input.cards_to_loc = {'AS': inst.input.cards_to_loc['AS']}
    return insts


def only_ace_dev():
    insts = cards_dev()
    for inst in insts:
        inst.input.cards_to_loc = {'AS': inst.input.cards_to_loc['AS']}
    return insts


def just_go_down():
    walls = [[1., 1., 1.],
             [1., 0., 1.],
             [1., 0., 1.],
             [1., 1., 1.]]
    cards_to_loc = {'AS': (2, 1)}
    return [Instance(input=build_world(walls, cards_to_loc), output=0)
            for _ in range(500)]


def dist():
    walls = np.ones(MAX_BOARD_SIZE)
    options = config.options()
    h, w = options.dist_num_rows, options.dist_num_cols
    walls[1:h + 1, 1:w + 1] = 0.
    player_loc = (1 + (h - 1) / 2, 1 + (w - 1) / 2)
    card_loc = (player_loc[0] + options.dist_offset_row,
                player_loc[1] + options.dist_offset_col)
    if not (1 <= card_loc[0] <= h and
            1 <= card_loc[1] <= w):
        raise ValueError('Card loc {} for dist is not in bounds {}; fix '
                         'dist_offset_[row,col].'.format(card_loc, walls.shape))
    cards_to_loc = {'AS': card_loc}
    return [Instance(input=build_world(walls, cards_to_loc, p1_loc=player_loc), output=0)
            for _ in range(500)]


def interpret_transcript(data):
    transcript, options = data
    pairs = world.event_world_pairs(transcript)
    for event, state in pairs:
        if event.action == cards.UTTERANCE:
            # Player 1 is always the listener, Player 2 is always the speaker
            state = state.line_of_sight(los=options.line_of_sight_p2,
                                        swap_players=(event.agent == cards.PLAYER1))
            yield Instance(input={'utt': ['<s>'] + event.parse_contents() + ['</s>'],
                                  'cards': world.build_cards_obs(state,
                                                                 options.line_of_sight),
                                  'walls': np.maximum(0.0, state.walls)},
                           output=state.__dict__)


def interpret(transcripts):
    options = config.options()
    return [inst for t in transcripts for inst in interpret_transcript((t, options))]


def interpret_train():
    insts = interpret(train_transcripts())
    rng.shuffle(insts)
    return insts


def interpret_dev():
    return interpret(dev_transcripts())


def interpret_test():
    return interpret(test_transcripts())


def get_location_utt(example, tokenizer):
    return tokenizer.tokenize(example['Text'])


def get_annotation_utt(example, tokenizer):
    return [example['Domain']] + example['Semantics'].split(';')


def location_insts(start=None, end=None, listener=True, get_utt=get_location_utt):
    insts = []
    tokenizer = cards.Tokenizer()
    with open('potts-wccfl30-supp/wccfl30-location-phrases.csv', 'r') as infile:
        reader = csv.DictReader(infile)
        examples = list(reader)[start:end]
    for example in examples:
        walls = world.walls_from_str(example['Map'])
        utt = ['<s>'] + get_utt(example, tokenizer) + ['</s>']
        loc = (int(example['X']), int(example['Y']))
        if listener:
            inst = Instance(input={'utt': utt, 'walls': walls}, output=loc)
        else:
            inst = Instance(input={'loc': loc, 'walls': walls}, output=utt)
        insts.append(inst)
    return insts


def location_l_all():
    return location_insts()


def location_l_train():
    return location_insts(end=400)


def location_l_dev():
    return location_insts(400, 500)


def location_l_test():
    return location_insts(start=500)


def location_s_train():
    return location_insts(end=400, listener=False)


def location_s_dev():
    return location_insts(400, 500, listener=False)


def location_s_test():
    return location_insts(start=500, listener=False)


def annotations_s_train():
    return location_insts(end=400, listener=False, get_utt=get_annotation_utt)


def annotations_s_dev():
    return location_insts(400, 500, listener=False, get_utt=get_annotation_utt)


def annotations_s_test():
    return location_insts(start=500, listener=False, get_utt=get_annotation_utt)


def location_speaker_prior(num_samples):
    '''
    Sample `num_samples` random worlds to feed to a location speaker.
    '''
    template = location_insts(end=1, listener=False)[0]
    insts = []
    walls = template.input['walls']
    locs = sample_valid_locs(walls, num_samples)
    for loc in locs:
        insts.append(Instance(input={'loc': loc, 'walls': walls}, output=['<s>', '</s>']))
    return insts


def location_listener_prior(num_samples):
    '''
    Sample `num_samples` random utterances to feed to a location listener.
    '''
    template = location_insts(end=1, listener=False)[0]
    insts = []
    walls = template.input['walls']
    utts = [inst.input['utt'] for inst in interpret_train()][:num_samples]
    for utt in utts:
        insts.append(Instance(input={'utt': utt, 'walls': walls}, output=(0, 0)))
    return insts


def sample_valid_locs(walls, num_samples):
    walls = np.array(walls)
    valid_locs = np.where(walls == 0)
    index_range = np.arange(valid_locs[0].shape[0])
    indices = np.random.choice(index_range, num_samples)
    return [(valid_locs[0][i], valid_locs[1][i]) for i in indices]


def samples_ls_to_ll():
    import learners
    options = config.options()
    learner = learners.new(options.sampler_model_learner)
    learner.load(options.sampler_model_load)
    inputs = location_speaker_prior(options.num_samples)
    samples = learner.predict(inputs, random=True)
    return [Instance(input={'utt': samp, 'walls': inp.input['walls']},
                     output=inp.input['loc'])
            for inp, samp in zip(inputs, samples)]


def samples_ll_to_ls():
    import learners
    options = config.options()
    learner = learners.new(options.sampler_model_learner)
    learner.load(options.sampler_model_load)
    inputs = location_listener_prior(options.num_samples)
    samples = learner.predict(inputs, random=True)
    return [Instance(input={'loc': samp, 'walls': inp.input['walls']},
                     output=inp.input['utt'])
            for inp, samp in zip(inputs, samples)]


DataSource = namedtuple('DataSource', ['train_data', 'test_data'])

SOURCES = {
    'cards_dev': DataSource(cards_train, cards_dev),
    'cards_test': DataSource(cards_train, cards_test),
    'single_loc': DataSource(single_loc_train, single_loc_dev),
    'only_ace': DataSource(single_loc_train, single_loc_dev),
    'just_go_down': DataSource(just_go_down, just_go_down),
    'dist': DataSource(dist, dist),
    'interpret_dev': DataSource(interpret_train, interpret_dev),
    'interpret_test': DataSource(interpret_train, interpret_test),
    'location': DataSource(location_l_all, location_l_all),
    'location_l_dev': DataSource(location_l_train, location_l_dev),
    'location_l_test': DataSource(location_l_train, location_l_test),
    'location_s_dev': DataSource(location_s_train, location_s_dev),
    'location_s_test': DataSource(location_s_train, location_s_test),
    'annotations_s_dev': DataSource(annotations_s_train, annotations_s_dev),
    'annotations_s_test': DataSource(annotations_s_train, annotations_s_test),
    'samples_ls_to_ll': DataSource(samples_ls_to_ll, location_l_dev),
    'samples_ll_to_ls': DataSource(samples_ll_to_ls, location_s_dev),
}
