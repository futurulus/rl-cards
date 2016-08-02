from collections import namedtuple

from stanza.research.instance import Instance
from stanza.research.rng import get_rng

from cards_cache import all_transcripts


rng = get_rng()


def cards_train():
    num_trans = len(all_transcripts())
    return [Instance(input=n, output=0)
            for n in range(num_trans)
            if n % 10 < 6]


def cards_dev():
    num_trans = len(all_transcripts())
    return [Instance(input=n, output=0)
            for n in range(num_trans)
            if n % 10 in (6, 7)]


def cards_test():
    num_trans = len(all_transcripts())
    return [Instance(input=n, output=0)
            for n in range(num_trans)
            if n % 10 in (8, 9)]


DataSource = namedtuple('DataSource', ['train_data', 'test_data'])

SOURCES = {
    'cards_dev': DataSource(cards_train, cards_dev),
    'cards_test': DataSource(cards_train, cards_test),
}
