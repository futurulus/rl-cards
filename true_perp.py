# Print the minimum perplexity and accuracy achievable by a
# "memorization" model on the training set.
#    $ python true_perp.py -R runs/rundir
import json
import numpy as np
from collections import Counter
from stanza.research import config


def print_true_perp():
    options = config.options(read=True)
    with config.open('data.train.jsons', 'r') as infile:
        data = [json.loads(line.strip()) for line in infile]
    print('# examples: {}'.format(len(data)))
    memory = get_memory(data)
    min_perp = minimum_perplexity(memory)
    print('minimum perplexity: {}'.format(min_perp))
    max_acc = maximum_accuracy(memory)
    print('maximum accuracy: {}'.format(max_acc))


def get_memory(data):
    memory = {}
    for inst in data:
        loc = tuple(inst['input']['loc'])
        if loc not in memory:
            memory[loc] = []
        memory[loc].append(tuple(inst['output']))
    return memory


def minimum_perplexity(memory):
    probs = []
    for v in memory.values():
        for utt in v:
            probs.append(v.count(utt) * 1.0 / len(v))
    return np.exp(-np.log(probs).mean())


def maximum_accuracy(memory):
    correct = 0
    total = 0
    for v in memory.values():
        memorized = Counter(v).most_common(1)[0][0]
        correct += v.count(memorized)
        total += len(v)
    return correct * 1.0 / total


if __name__ == '__main__':
    print_true_perp()
