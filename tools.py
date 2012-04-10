""" Just some tools to help with the neural network trainer.
"""

import random


def zero(l, start, end):
    """ zeros part of a list.
    """
    l[start:end + 1] = [0] * ((end + 1) - start)
    return l

def normalise(value):
    if value >= 0.5:
        return 1
    return 0

def mutate(l, freeoutputs, statesets, seed=None, momentum=0.75):

    if seed is not None:
        random.seed(seed)

    for i in freeoutputs:
        if random.random() > momentum:
            l[i] = random.random()

    for ss,sf in statesets:
        if random.random() > momentum:
            l = zero(l, ss, sf)
            l[random.randint(ss, sf)] = random.random()

    return l

def notinranges(length, ranges):
    """ Returns list of indexes between 0 and length where indexes not in
        list of ranges passed in.
    """
    indexes_in_a_range = []
    for rs,rf in ranges:
        indexes_in_a_range.extend(range(rs, rf + 1))
    return [i for i in range(length) if i not in indexes_in_a_range]
