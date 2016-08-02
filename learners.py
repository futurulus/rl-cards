from baseline import RandomLearner, SearcherLearner, OracleLearner


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
}
