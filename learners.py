from baseline import RandomLearner, SearcherLearner, OracleLearner
from rl_learner import KarpathyPGLearner, RLListenerLearner
from reflex import UniformListener, ReflexListener, FactoredReflexListener
from reflex import LocationListener, LocationSpeaker, SmoothedLocationSpeaker
from reflex import ContrastiveSmoothedLocationSpeaker


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
    'RLL': RLListenerLearner,
    'UniformL': UniformListener,
    'ReflexL': ReflexListener,
    'FactoredReflexL': FactoredReflexListener,
    'Location': LocationListener,
    'LocationL': LocationListener,
    'LocationS': LocationSpeaker,
    'SmoothedLocationS': SmoothedLocationSpeaker,
    'ContrastiveLocationS': ContrastiveSmoothedLocationSpeaker,
}
