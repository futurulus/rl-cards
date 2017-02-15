from baseline import RandomLearner, SearcherLearner, OracleLearner
from rl_learner import KarpathyPGLearner
from reflex import UniformListener, ReflexListener


def new(key):
    '''
    Construct a new learner with the class named by `key`. A list
    of available learners is in the dictionary `LEARNERS`.
    '''
    return LEARNERS[key]()


LEARNERS = {
    'Random': RandomLearner,
    'Searcher': SearcherLearner,
    'Oracle': OracleLearner,
    'KarpathyPG': KarpathyPGLearner,
    'UniformL': UniformListener,
    'ReflexL': ReflexListener,
}
