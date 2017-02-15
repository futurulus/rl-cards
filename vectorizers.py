import numpy as np
from collections import Sequence, Counter

from stanza.research.rng import get_rng

rng = get_rng()


class SymbolVectorizer(object):
    '''
    Maps symbols from an alphabet/vocabulary of indefinite size to and from
    sequential integer ids.

    >>> vec = SymbolVectorizer()
    >>> vec.add_all(['larry', 'moe', 'larry', 'curly', 'moe'])
    >>> vec.vectorize_all(['curly', 'larry', 'moe', 'pikachu'])
    array([3, 1, 2, 0], dtype=int32)
    >>> vec.unvectorize_all([3, 3, 2])
    ['curly', 'curly', 'moe']
    '''
    def __init__(self):
        self.tokens = []
        self.token_indices = {}
        self.indices_token = {}
        self.add('<unk>')

    @property
    def num_types(self):
        return len(self.tokens)

    def add_all(self, symbols):
        for sym in symbols:
            self.add(sym)

    def add(self, symbol):
        if symbol not in self.token_indices:
            self.token_indices[symbol] = len(self.tokens)
            self.indices_token[len(self.tokens)] = symbol
            self.tokens.append(symbol)

    def vectorize(self, symbol):
        return (self.token_indices[symbol] if symbol in self.token_indices
                else self.token_indices['<unk>'])

    def vectorize_all(self, symbols):
        return np.array([self.vectorize(sym) for sym in symbols], dtype=np.int32)

    def unvectorize(self, index):
        return self.indices_token[index]

    def unvectorize_all(self, array):
        if hasattr(array, 'tolist'):
            array = array.tolist()
        return [self.unvectorize(elem) for elem in array]


class SequenceVectorizer(object):
    '''
    Maps sequences of symbols from an alphabet/vocabulary of indefinite size
    to and from sequential integer ids.

    >>> vec = SequenceVectorizer()
    >>> vec.add_all([['the', 'flat', 'cat', '</s>', '</s>'], ['the', 'cat', 'in', 'the', 'hat']])
    >>> vec.vectorize_all([['in', 'the', 'cat', 'flat', '</s>'],
    ...                    ['the', 'cat', 'sat', '</s>', '</s>']])
    array([[5, 1, 3, 2, 4],
           [1, 3, 0, 4, 4]], dtype=int32)
    >>> vec.unvectorize_all([[1, 3, 0, 5, 1], [1, 2, 3, 6, 4]])
    [['the', 'cat', '<unk>', 'in', 'the'], ['the', 'flat', 'cat', 'hat', '</s>']]
    '''
    def __init__(self, unk_threshold=0):
        self.tokens = []
        self.token_indices = {}
        self.indices_token = {}
        self.counts = Counter()
        self.max_len = 0
        self.unk_threshold = unk_threshold
        self.add(['<unk>'] * (unk_threshold + 1))

    @property
    def num_types(self):
        return len(self.tokens)

    def add_all(self, sequences):
        for seq in sequences:
            self.add(seq)

    def add(self, sequence):
        self.max_len = max(self.max_len, len(sequence))
        self.counts.update(sequence)
        for token in sequence:
            if token not in self.token_indices and self.counts[token] > self.unk_threshold:
                self.token_indices[token] = len(self.tokens)
                self.indices_token[len(self.tokens)] = token
                self.tokens.append(token)

    def unk_replace(self, sequence):
        return [(token if token in self.token_indices else '<unk>')
                for token in sequence]

    def unk_replace_all(self, sequences):
        return [self.unk_replace(s) for s in sequences]

    def vectorize(self, sequence):
        sequence = self.pad(sequence)
        return np.array([(self.token_indices[token] if token in self.token_indices
                          else self.token_indices['<unk>'])
                         for token in sequence], dtype=np.int32)

    def vectorize_all(self, sequences):
        return np.array([self.vectorize(seq) for seq in sequences], dtype=np.int32)

    def unvectorize(self, array):
        if hasattr(array, 'tolist'):
            array = array.tolist()
        return [(self.unvectorize(elem) if isinstance(elem, Sequence)
                 else self.indices_token[elem])
                for elem in array]

    def unvectorize_all(self, sequences):
        # unvectorize already accepts sequences of sequences.
        return self.unvectorize(sequences)

    def pad(self, sequence):
        return sequence[:self.max_len] + ['<MASK>'] * (self.max_len - len(sequence))


def strip_invalid_tokens(sentence):
    good_tokens = [t for t in sentence if t not in ('<s>', '<MASK>')]
    if '</s>' in good_tokens:
        end_pos = good_tokens.index('</s>')
        good_tokens = good_tokens[:end_pos]
    return good_tokens
