#!/usr/bin/env python3
# Copyright 2021 Feras Baig

import os
import sys

import keras
import tensorflow as tf

from prediction.network import Network
from hyperproperties import HyperProperties, RobustnessProperties, SafetyProperties, LivenessProperties
from bound_propagation import BoundPropagation

class Specification():
    """
        Description: Core Formal Specifications for Deep Convolutional Neural Network. Write and aggregate all specifications and sub-nodes in verification node to compute on network during its training and testing
        Args:
            self.bound_propagation = BoundPropagation(): store algorithm for bound propagation
            self._robustness_properties = RobustnessProperties(): store initialized robustness properties and compute specifications for robustness verification
            self._safety_properties = SafetyProperties(): : store initialized safety properties and compute specifications for safety verification
            self._liveness_properties = LivenessProperties(): : store initialized liveness properties and compute specifications for liveness verification

        Returns:
        Type: Specification ( object that stores metrics for all computations for safety verification, robustness, and liveness properties)

        Raises:
        References:
    """

    def __init__(self):
        # store hyperproperty objects and functions to compute verification algorithms
        self.bound_propagation = BoundPropagation()
        self._robustness_properties = RobustnessProperties()
        self._safety_properties = SafetyProperties()
        self._liveness_properties = LivenessProperties()
        self.verificationState = False # set to True when all trace properties have been checked


    def main(self):
        '''
            Description: 
            Args:
            Returns:
            Raises:
            References:
            Examples:
        '''
        raise NotImplementedError


class CheckTraceData():
    '''
        Description: 
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''  
    raise NotImplementedError

class SafetyTrace():
    '''
        Description: 
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''
    raise NotImplementedError

class RobustnessTrace():
    '''
        Description: 
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''
    raise NotImplementedError


if __name__ == '__main__':
    Specification()

