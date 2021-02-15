#!/usr/bin/env python3
# Copyright 2021 Feras Baig

'''
Utility to convert symbolic representation into formal logic statement and specification problem.
'''

import os
import random
import sys
import time

import keras
import tensorflow as tf
from prediction.network import Network


class SymbolicRepresentation():
    """Symbolic State Representation for Finite-State Abstraction of Neural Network Node"""
    def __init__(self):
        self.network = Network()
        # store variable access points here for symbolic representation, and make sure all values are initialized with floating points

        # store variables to symbolically represent variables of the network's state


    raise NotImplementedError


if __name__ == '__main__':
    SymbolicRepresentation()
