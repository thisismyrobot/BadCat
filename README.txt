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
   * OK, but a bit brute-force.

Requirements
============

This has been developed/tested on:

 * Ubuntu 12.04
 * Python 2.7

The neural network used for trainer testing is from:
http://code.google.com/p/neurolab/

For tests to pass, requires the "robosim" module from
https://github.com/thisismyrobot/RoboSim
