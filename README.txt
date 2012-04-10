The Gist
========

Finally dev'ing a hunch - can you train a Neural Network using just "good" or
"bad" as feedback?

Progress
========

Trying out each of the following techniques to see which seems to work ok:

 * If bad, train with inversion of last output.
   * Flawed - works with XOR but not with anything more complex.
 * If bad, mutate last output until good and then train.
   * Current status - a bit brute-force so will explore gentler versions next.

Requirements
============

This has been developed/tested on:

 * Ubuntu 12.04
 * Python 2.7

The neural network used for trainer testing is from:
http://code.google.com/p/neurolab/

For tests to pass, requires the "robosim" module from
https://raw.github.com/thisismyrobot/RoboSim/master/robosim.py

Tests
=====

Will pass most of the time, but the neurolab library does not have a seedable
random so results can vary. Running the tests a number of times (2-5) usually
results in a pass.
