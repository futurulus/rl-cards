from collections import namedtuple

from stanza.research.instance import Instance
from stanza.research.rng import get_rng

from world import CardsWorld
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


DataSource = namedtuple('DataSource', ['train_data', 'test_data'])

SOURCES = {
    'cards_dev': DataSource(cards_train, cards_dev),
    'cards_test': DataSource(cards_train, cards_test),
}
