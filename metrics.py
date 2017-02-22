import numpy as np

from stanza.research.metrics import log_likelihood
from stanza.research.metrics import log_likelihood_bits, accuracy, perplexity  # NOQA


def reward(*args, **kwargs):
    return log_likelihood(*args, **kwargs)


def loc_perplexity(eval_data, predictions, scores, learner='ignored'):
    return np.exp(-np.array(scores) / 53.0).tolist()


def loc_accuracy(eval_data, predictions, scores, learner='ignored'):
    return [loc_accuracy_inst(inst, pred) for inst, pred in zip(eval_data, predictions)]


def loc_accuracy_inst(inst, pred):
    score = 0.0
    cards_to_loc = inst.output['cards_to_loc']
    for k in cards_to_loc:
        if cards_to_loc[k] == pred['cards_to_loc'][k]:
            score += 1.0
    if inst.output['p2_loc'] == pred['p2_loc']:
        score += 1.0
    total = len(cards_to_loc) + 1.0
    return score / total


METRICS = {
    name: globals()[name]
    for name in dir()
    if (name not in ['np']
        and not name.startswith('_'))
}
