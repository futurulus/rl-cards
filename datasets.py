from collections import namedtuple

from stanza.research.instance import Instance
from stanza.research.rng import get_rng

from world import CardsWorld, build_world
from cards_cache import all_transcripts


rng = get_rng()


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


def just_go_down():
    walls = [[1., 1., 1.],
             [1., 0., 1.],
             [1., 0., 1.],
             [1., 1., 1.]]
    cards_to_loc = {'AS': (2, 1)}
    return [Instance(input=build_world(walls, cards_to_loc), output=0)
            for _ in range(500)]


DataSource = namedtuple('DataSource', ['train_data', 'test_data'])

SOURCES = {
    'cards_dev': DataSource(cards_train, cards_dev),
    'cards_test': DataSource(cards_train, cards_test),
    'single_loc': DataSource(single_loc_train, single_loc_dev),
    'just_go_down': DataSource(just_go_down, just_go_down),
}
