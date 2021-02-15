#!/usr/bin/env python3
# Copyright 2021 Feras Baig

import os
import sys

import keras
import tensorflow as tf

from prediction.network import Network
from hyperproperties import HyperProperties, RobustnessProperties, SafetyProperties
from bound_propagation import BoundPropagation

'''
Description: 
    - Given defined trace properties in `verification.hyperproperties`, compute and iterate over each trace property and track if the computed (symbolic) state abstraction produces a counter-example of the property trace. 
    - Note that this file is meant to store trace property checkers that adhere to constraints of temporal logics, I am simply isolating the verification / checkers themselves e.g. reachability.py, symbolic_interval_analysis.py
'''
import os
import time
import random
import torch
from torch import nn

from prediction.network import Network
from hyperproperties import HyperProperties, RobustnessProperties, SafetyProperties
from bound_propagation import BoundPropagation


class STLSpecification(Network):
    '''
        Description: Compute property inferencing and checking given formal specifications in `hyperproperties`, which stores all of the trace properties, and `verification.stl` will process the functions and variables of the hyperproperties, which will have computed the data that accesses the network state to model the network state to compute all the verifications, for which reachability sets will be computed in `symbolic_representation.py` and the network state and the computational model for the network state for the specifications to be computed will be stored in hyperproperties. The `bound_propagation` file is meant to compute the upper and lower bounds for the reachability problem, adjacent to the other processes.
        Args:
            - self.robustness = RobustnessProperties()
            - self.safety = SafetyProperties()
            - self.liveness = LivenessProperties()
        Returns:
        Raises:
        References:
        Examples:
    '''

    def __init__(self):
        self.robustness = RobustnessProperties()
        self.safety = SafetyProperties()
        self.liveness = LivenessProperties()



class Trace():
    '''
        Description: 
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''
    def __init__(self):
        self.l2norm = 0
        self.upperBound = {}
        self.lowerBound = {}
        self.reachableSets = {}

    @staticmethod
    def check_trace():
        raise NotImplementedError


class STLCheckTraceData():
    '''
        Description: 
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''
    raise NotImplementedError

class STLSafetyTrace():
    '''
        Description: Store safety properties that will be checked with bounded model checking. Note that each function of safety trace will be its own safety trace property.
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''
    raise NotImplementedError

class STLRobustnessTrace():
    '''
        Description: 
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''
    raise NotImplementedError


class STLSolver():
    '''
        Description: Compute BMC (Bounded Model Checking) to Compute Violation of Signal-Temporal Specifications Given Temporal Bounds. Specifically, iter in range(H) for H = {R, S} for R and S are the subset lists of trace properties for R: robustness, S: safety
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''
    raise NotImplementedError

if __name__ == '__main__':
    Trace.check_trace()

