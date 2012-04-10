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

import collections
import tools


class Trainer(object):
    """ Trains a neural network using "good" or "bad" feedback.

        Order of ops:

            # get the output for a particularly sensed input
            getoutput()

            # if trying the output is "bad", repeatedly call bad() to generate
            # a new output until it results in "good"
            [bad()]

            # call good() to save that output for training
            good()

    """

    def __init__(self, nn, memorysize, statesets):
        self.nn = nn

        # (input, target) store
        self._memory = collections.deque([], memorysize)
        self._statesets = statesets

        self._lastinput = [0] * self.nn.ci
        self._lastoutput = [0] * self.nn.co

    @property
    def _freeoutputs(self):
        stateoutputs = []
        for sset in self._statesets:
            stateoutputs.extend(range(sset[0], sset[1] + 1))
        return [i for i in range(self.nn.co) if i not in stateoutputs]

    def getoutput(self, input=None):
        if input is None:
            input = self._lastinput
        else:
            self._lastinput = input
        self._lastoutput = self.nn.step(input)
        return tuple(map(tools.normalise, self._lastoutput))

    def good(self, epochs=500):
        """ Add the input-output mapping to a memory buffer for training, do
            the training
        """
        self._memory.append((self._lastinput,
                             tuple(map(tools.normalise, self._lastoutput))))

        self.nn.train([i for i, o in self._memory],
                      [o for i, o in self._memory],
                      show=None, epochs=epochs)

    def bad(self):
        tools.mutate(self._lastoutput, self._freeoutputs, self._statesets)
        return tuple(map(tools.normalise, self._lastoutput))
