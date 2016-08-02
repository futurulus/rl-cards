from stanza.research.metrics import log_likelihood


def reward(*args, **kwargs):
    return log_likelihood(*args, **kwargs)


METRICS = {
    name: globals()[name]
    for name in dir()
    if (name not in ['np']
        and not name.startswith('_'))
}
