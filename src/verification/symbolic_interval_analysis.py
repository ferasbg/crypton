#!/usr/bin/env python3
# Copyright 2021 Feras Baig

import os
import sys

import keras
import tensorflow as tf
from prediction.network import Network
from prediction.network_convert import BoundedNetwork

from bound_propagation import BoundPropagation
from hyperproperties import RobustnessProperties, SafetyProperties


class PublicSymbolicInterval():
    '''
        Description:

        Args: BoundedNetwork

        Returns:

        Raises:

        References:

        Examples:

    '''
    raise NotImplementedError


class MPCSymbolicInterval(PublicSymbolicInterval):
    '''
        Description:

        Args:

        Returns:

        Raises:

        References:

        Examples:

    '''
    raise NotImplementedError


class SymbolicIntervalAnalysis(PublicSymbolicInterval):
    '''
        Description: Compute Symbolic Interval Analysis with differentiable lower and upper bounds of tf.keras.model.ReLU.getLayerState(). 

        Args:
            - self.reluState = prediction.network_convert.BoundedNetwork.getLayerState()
            - self.state_representation
            - self.upperBound
            - self.lowerBound

        Returns:

        Raises:

        References:
            - https://github.com/tcwangshiqi-columbia/symbolic_interval/tree/master/symbolic_interval
            - https://github.com/tcwangshiqi-columbia/symbolic_interval/blob/a008fbb54d04b3a3005cc3967b858c35a3fcf3dd/symbolic_interval/symbolic_network.py#L29:7
            - https://github.com/tcwangshiqi-columbia/symbolic_interval/blob/master/test.py

    '''

    raise NotImplementedError


class IntervalIterativeRefinement():
    '''
        Description:
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''
    raise NotImplementedError

class MPCIntervalIterativeRefinement(IntervalIterativeRefinement):
    '''
        Description:
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''
    raise NotImplementedError

class SymbolicIntervalSplitting():
    '''
        Description:
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''
    raise NotImplementedError


class MPCSymbolicIntervalSplitting(SymbolicIntervalSplitting):
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
    # Will most likely need to compartmentalize and organize code with respect to classes, but for now this is fine.
    # add params that are necessary for each Object for Symbolic Interval Analysis of Network for Reachability + Bound Propagation
    PublicSymbolicInterval()
    SymbolicIntervalAnalysis()
