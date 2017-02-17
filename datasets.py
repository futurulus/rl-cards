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
                    help='Maximum distance (Manhattan distance) from the player that a card is '
                         'visible.')
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


def interpret(transcripts):
    options = config.options()
    insts = []
    for i, trans in enumerate(transcripts):
        if i % 100 == 0:
            print(i)
        pairs = world.event_world_pairs(trans)
        for event, state in pairs:
            if event.action == cards.UTTERANCE:
                # Player 1 is always the listener, Player 2 is always the speaker
                if event.agent == cards.PLAYER2:
                    state = state.swap_players()
                insts.append(Instance(input={'utt': ['<s>'] + event.parse_contents() + ['</s>'],
                                             'cards': world.build_cards_obs(state,
                                                                            options.line_of_sight),
                                             'walls': np.maximum(0.0, state.walls)},
                                      output=state.__dict__))
    return insts


def interpret_train():
    insts = interpret(train_transcripts())
    rng.shuffle(insts)
    return insts


def interpret_dev():
    return interpret(dev_transcripts())


def interpret_test():
    return interpret(test_transcripts())


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
}
