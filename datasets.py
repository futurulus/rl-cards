from collections import namedtuple
import numpy as np

from stanza.research.rng import config
from stanza.research.instance import Instance
from stanza.research.rng import get_rng

from world import CardsWorld, build_world, MAX_BOARD_SIZE
from cards_cache import all_transcripts


rng = get_rng()

parser = config.get_options_parser()
parser.add_argument('--dist_offset_row', type=int, default=1,
                    help='Which row to place the ace of spades in '
                         '(relative to player position). '
                         'Used only in "dist" data_source.')
parser.add_argument('--dist_offset_col', type=int, default=0,
                    help='Which column to place the ace of spades in '
                         '(relative to player position). '
                         'Used only in "dist" data_source.')


def cards_train():
    insts = [Instance(input=CardsWorld(trans), output=0)
             for n, trans in enumerate(all_transcripts())
             if n % 10 < 6]
    rng.shuffle(insts)
    return insts


def cards_dev():
    return [Instance(input=CardsWorld(trans), output=0)
            for n, trans in enumerate(all_transcripts())
            if n % 10 in (6, 7)]


def cards_test():
    return [Instance(input=CardsWorld(trans), output=0)
            for n, trans in enumerate(all_transcripts())
            if n % 10 in (8, 9)]


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
    walls = np.zeros(MAX_BOARD_SIZE)
    walls[0, :] = 1.
    walls[:, 0] = 1.
    walls[-1, :] = 1.
    walls[:, -1] = 1.
    player_loc = (walls.shape[0] / 2, walls.shape[1] / 2)
    options = config.options()
    card_loc = (player_loc[0] + options.dist_offset_row,
                player_loc[1] + options.dist_offset_col)
    if not (1 <= card_loc[0] < walls.shape[0] - 1 and
            1 <= card_loc[1] < walls.shape[1] - 1):
        raise ValueError('Card loc {} for dist is not in bounds {}; fix '
                         'dist_offset_[row,col].'.format(card_loc, walls.shape))
    cards_to_loc = {'AS': card_loc}
    return [Instance(input=build_world(walls, cards_to_loc, p1_loc=player_loc), output=0)
            for _ in range(500)]


DataSource = namedtuple('DataSource', ['train_data', 'test_data'])

SOURCES = {
    'cards_dev': DataSource(cards_train, cards_dev),
    'cards_test': DataSource(cards_train, cards_test),
    'single_loc': DataSource(single_loc_train, single_loc_dev),
    'only_ace': DataSource(single_loc_train, single_loc_dev),
    'just_go_down': DataSource(just_go_down, just_go_down),
    'dist': DataSource(dist, dist),
}
