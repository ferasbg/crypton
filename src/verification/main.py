#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig

import os
import pickle
import random
import sys

import keras
import tensorflow as tf

from specification import CheckTraceData, RobustnessTrace, SafetyTrace


class BoundedNetworkSolver():
    '''
        Description: Compute BMC (Incremental, Parameterized Bounded Model Checking) to Compute Violation of Signal-Temporal Specifications Given Temporal Bounds for Network. In other words, use constraint-satisfaction solver to evaluate reachable states concerned with (e.g. output layer, dense) to process bounded network state abstraction to compute satisfiability for each trace property.
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''
    raise NotImplementedError


class BoundedMPCNetworkSolver(BoundedNetworkSolver):
    '''
        Description: Compute BMC (Bounded Model Checking) to Compute Violation of Signal-Temporal Specifications Given Temporal Bounds for MPCNetwork
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''
    raise NotImplementedError


class VerifyTrace():
    '''
        Description: Given Computed Formal State Representation (e.g. BMC, STL, SymbolicInterval, BoundPropagation), Compute Probabilistic / Boolean Satisfiability Iterating Over All Traces in SafetyTrace, RobustnessTrace
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''

    @staticmethod
    def verify_trace():
        # get all property specifications
        # check the state of the specification to check if trace property has been satisfied
        if (RobustnessTrace.affine_abstract_output_vector_state == True and RobustnessTrace.adversarial_sample_created == False):
            return True
        
        else:
            return False
