#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig
import numpy as np
import matplotlib as plt
import keras
import tensorflow as tf
import os
import sys
import random
import pickle

from prediction.network import Network


class BoundPropagation():
    """
    Define and compute bound propagation to define constraints for trace properties to then compute violations of the trace properties for t \epsilon H for H = {safety, robustness, liveness}. Make formal guarantees with upper and lower bounds that maintain reliability of the network's behavior (optimization technique), formalizing it into constraint satisfaction problem.
    Args:
        - self.state_size: symbolic abstraction store
        - self.upper_bound: upper bounds for each layer for symbolic interval
        - self.lower_bound: lower bounds for each layer for symbolic interval analysis, based on state, specification is met / not met

    Raises:

    Returns:

    References:
        - https://github.com/deepmind/interval-bound-propagation/

    """

    def __init__(self, *args, **kwargs):
        self.epsilon = 0.0003
        self.upperBound = np.randn(128, 20)
        self.lowerBound = np.randn(128, 20)







@abstractclass
class Bounds(BoundPropagation):
    """Compute bounds given ReLU State of Neural Network for Robustness Verification (property inference and checking)"""
    def __init__(self):
        super(Bounds, self).__init__()
        # signed 8-bit ints to signify lipschitz constant
        self.upperBound = 0
        self.lowerBound = 0



if __name__ == '__main__':
    BoundPropagation()
    Bounds()

