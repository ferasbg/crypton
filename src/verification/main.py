#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig

import os
import pickle
import random
import sys

import keras
import tensorflow as tf

from specification import CheckTraceData, RobustnessTrace 


class BoundedNetworkSolver():
    '''
        Description: Compute BMC (Incremental, Parameterized Bounded Model Checking) to Compute Violation of Signal-Temporal Specifications Given Temporal Bounds for Network. 
        
        Note, we are initializing our constraint-satisfaction problem, our prepositional modal logic formula to assert robust output states after network state-transition T(x) for P(x,y) => Q(x,y) | x := y or input-output states match and are thus robust and thus correctness is proven. Negate non-robust output states and consider them in violation of the constraint specified.        
        
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''

    def symbolically_encode_network(self, network):
        """Convert convolutional network state into prepositional formula given the constraint-satisfaction problem that it will be evaluated against with initialize_constraint_satisfaction_formula()"""
        raise NotImplementedError

    def get_relevant_reachable_states(self, network):
        raise NotImplementedError

    def initialize_constraint_satisfaction_formula(self):
        raise NotImplementedError

    def bmc_to_propositional_satisfiability(self):
        """Synthesize logical formula translated through encoding convolutional network, using symbolically_encode_network() and initialize_constraint_satisfaction_formula() for s.e.n is passed to i.c.s.f., and then parameter synthesis will evaluate or iterate over reachable states only to then update the state of the respective trace property for the SMT model checker."""
        raise NotImplementedError

    def traverse_robust_network_state_transitions(self):
        raise NotImplementedError

    def smt_solver_trace_check(self):
        raise NotImplementedError



class BoundedCryptoNetworkSolver(BoundedNetworkSolver):
    '''
        Description: Compute BMC (Bounded Model Checking) to Compute Violation of Signal-Temporal Specifications Given Temporal Bounds for CryptoNetwork.
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''

    def reconstruct_secrets(self):
        raise NotImplementedError
        

class VerifyTrace():
    '''
        Description: Given Computed Formal State Representation (e.g. BMC, STL, SymbolicInterval, BoundPropagation), Compute Probabilistic / Boolean Satisfiability Iterating Over All Traces in RobustnessTrace
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
