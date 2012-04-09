import collections
import random


class Trainer(object):
    """ Order of ops:
         1. __init__()
         2. getoutput()
         3. good() or bad()
    """

    def __init__(self, nn, memorysize, statesets, seed=None):
        if seed is None:
            random.seed()
        else:
            random.seed(seed)

        self.nn = nn

        # (input, target) store
        self.memory = collections.deque([], memorysize)
        self.statesets = statesets

        self._lastinput = [0] * self.nn.ci
        self._lastoutput = [0] * self.nn.co

    @property
    def _freeoutputs(self):
        stateoutputs = []
        for sset in self.statesets:
            stateoutputs.extend(range(sset[0], sset[1] + 1))
        return [i for i in range(self.nn.co) if i not in stateoutputs]

    @staticmethod
    def _zero(l, start, end):
        l[start:end + 1] = [0] * ((end + 1) - start)

    def _mutate(self):
        for i in self._freeoutputs:
            if random.random() > 0.75:
                self._lastoutput[i] = random.random()
        for ss,sf in self.statesets:
            if random.random() > 0.75:
                self._zero(self._lastoutput, ss, sf)
                self._lastoutput[random.randint(ss, sf)] = random.random()

    @staticmethod
    def _normalise(value):
        if value >= 0.5:
            return 1
        return 0

    def getoutput(self, input=None):
        if input is None:
            input = self._lastinput
        else:
            self._lastinput = input
        self._lastoutput = self.nn.step(input)
        return tuple(map(Trainer._normalise, self._lastoutput))

    def good(self, epochs=500):
        """ Add the input-output mapping to a memory buffer for training, do
            the training
        """
        self.memory.append((self._lastinput,
                            tuple(map(Trainer._normalise, self._lastoutput))))

        self.nn.train([i for i, o in self.memory],
                      [o for i, o in self.memory],
                      show=None, epochs=epochs)

    def bad(self):
        self._mutate()
        return tuple(map(Trainer._normalise, self._lastoutput))
