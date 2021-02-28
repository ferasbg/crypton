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

    def symbolically_encode_network(self, network):
        raise NotImplementedError

    def get_relevant_reachable_states(self, network):
        raise NotImplementedError

    def initialize_constraint_satisfaction_formula(self):
        raise NotImplementedError

    def getRobustnessTrace(self):
        # iterate over all traces and store in tuple or array to then evaluate each trace element
        adversarial_example = RobustnessTrace.adversarial_example_not_created()
        return adversarial_example # return array of all defined robustness trace properties, so define robustness trace properties given attack_types are fgsm, brightness_norm, adversarial_perturbation_for_adversarial_example_generation, projected_gradient_descent_attack_to_maximize_loss
        

class BoundedMPCNetworkSolver(BoundedNetworkSolver):
    '''
        Description: Compute BMC (Bounded Model Checking) to Compute Violation of Signal-Temporal Specifications Given Temporal Bounds for MPCNetwork
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''

    

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
        if (RobustnessTrace.adversarial_sample_created == False):
            return True
        
        else:
            return False
