#!/usr/bin/env python3
# Copyright 2021 Feras Baig
import os
import pickle
import random
import sys

import keras
import matplotlib as plt
import numpy as np
import tensorflow as tf
from prediction.network import Network

from verification.hyperproperties import RobustnessProperties, SafetyProperties
from verification.symbolic_interval_analysis import (
    IntervalIterativeRefinement, PublicSymbolicInterval,
    SymbolicIntervalAnalysis, SymbolicIntervalSplitting)


class ReachableSet():
    '''
    Description:
    Args:
    Returns:
    Raises:
    References:
    Examples:

    '''
    raise NotImplementedError

class ReachabilityAnalysis():

    '''
    Compute Reachability Analysis with Symbolic Intervals on Deep Convolutional Neural Network

    Args (Properties):
        - self.reach_point: defined timestep and object to store network state given initialized metadata and properties of neural network
        - self.reach_option: analyzing different subsets of object state at future timesteps
        - self.reachSet: reachable set before pixel classification layer
        - self.ground_truth: store matrix with tuples (ex: ['image[0][0]', 'road']) that store label for each pixel of 224x224 image
        - self.reach_time: store computation time to compare against other verification tasks


    Returns:
        Type: ReachabilityAnalysis

    Raises:
        NetworkVariableNULLError, ReachableSetTypeDefError

    References:
        - https://arxiv.org/abs/1805.02242

    '''
    def __init__(self):
        self.reach_point = []
        self.reach_option = []
        self.reachSet = []
        self.ground_truth = []
        self.reach_time = 0
        self.inputQuery = []


    def parse(self):
        raise NotImplementedError


if __name__ == '__main__':
    ReachabilityAnalysis()
