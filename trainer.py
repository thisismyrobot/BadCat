# Copyright 2012 Robert Wallhead
# robert@thisismyrobot.com
# <http://thisismyrobot.com>
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from abc import ABCMeta, abstractmethod, abstractproperty
import collections
import tools


class LearningToolAdapter:
    """ Allows learning tools to be adapted to the interface that the trainer
        expects. The learning tool object to be adapted is retrieved by
        self._lt.
    """
    __metaclass__ = ABCMeta

    def __init__(self, lt):
        self._lt = lt

    @abstractproperty
    def ci(self):
        """ Returns a count of input points
        """
        pass

    @abstractproperty
    def co(self):
        """ Returns a count of output points
        """
        pass

    @abstractmethod
    def learn(self, inputs, outputs):
        """ Given lists of inputs and outputs, train away.
        """
        pass

    @abstractmethod
    def process(self, input):
        """ Given an input list, return an output list.
        """
        pass


class Trainer(object):
    """ Trains a learning tool (eg neural network) using "good" or "bad"
        feedback.

        Order of ops:

            # get the output for a particularly sensed input
            getoutput()

            # if trying the output is "bad", repeatedly call bad() to generate
            # a new output until it results in "good"
            [bad()]

            # call good() to save that output and train with it.
            good()

    """
    def __init__(self, lt, memorysize, statesets, mutatemomentum=0.75):
        """ Creates and sets up a trainer.

            Arguments:
             * lt = a learning tool object with the interface of
               LearningToolAdapter.
             * the size of the memory of recent "learnings".
             * the set of ranges of points that only one can fire at once in.

        """
        if not isinstance(lt, LearningToolAdapter):
            raise Exception("Learning tools must be wrapped in LearningToolAdapter instances")
        self._lt = lt

        self._mutatemomentum = self._MUTATEMOMENTUMSTART = 0.75

        self._memory = collections.deque([], memorysize)
        self._statesets = statesets
        self._freeoutputs = tools.notinranges(self._lt.co, self._statesets)

        self._lastinput = [0] * self._lt.ci
        self._lastoutput = [0] * self._lt.co

    def _valid(self, output):
        """ Is an output valid, considering the statesets. For each set, if
            more than one value is not equal to zero, the output is invalid.
        """
        for ss, sf in self._statesets:
            l = map(tools.normalise, output[ss:sf + 1])
            if sum(l) > 1:
                return False
        return True

    def getoutput(self, input=None):
        if input is not None:
            self._lastinput = input

        self._lastoutput = self._lt.process(self._lastinput)

        while not self._valid(self._lastoutput):
            self.bad()

        return tuple(map(tools.normalise, self._lastoutput))

    def good(self):
        """ Add the input-output mapping to a memory buffer for training, do
            the training
        """
        self._mutatemomentum = self._MUTATEMOMENTUMSTART
        io = (self._lastinput, tuple(map(tools.normalise, self._lastoutput)))
        if io not in self._memory:
            self._memory.append(io)
            self._lt.learn([i for i, o in self._memory],
                           [o for i, o in self._memory])

    def bad(self, mutatestep=0.1):
        self._lastoutput = tools.mutate(self._lastoutput, self._freeoutputs,
                                        self._statesets, self._mutatemomentum)
        if self._mutatemomentum >= mutatestep:
            self._mutatemomentum -= mutatestep
        return tuple(map(tools.normalise, self._lastoutput))
