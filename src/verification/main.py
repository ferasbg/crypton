#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig

import os
import pickle
import random
import sys

import keras
import tensorflow as tf

from verification.specification import RobustnessTrace 
from adversarial.main import Adversarial


class BoundedNetworkSolver():
    '''
        Description: Compute BMC (Incremental, Parameterized Bounded Model Checking) to Compute Violation of Signal-Temporal Specifications Given Temporal Bounds for Network. 
        
        Note, we are initializing our constraint-satisfaction problem, our prepositional modal logic formula to assert robust output states after network state-transition T(x) for P(x,y) => Q(x,y) | x := y or input-output states match and are thus robust and thus correctness is proven. Negate non-robust output states and consider them in violation of the constraint specified.        

        Note that this logic proves partial program correctness, since it's evaluating the input-output relations under the constraint of an adversarial attack on the network's inputs.

        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''

    @staticmethod
    def symbolically_encode_network(network_precondition, network_postcondition):
        """Convert convolutional network state into propositional formula given the constraint-satisfaction problem defined in trace.
        
        Store some static variable state to indicate satisfiability in order to update the respective trace state to verify against specification given model checker.
        
        Args:
            - network_precondition | Type := image_label_element isinstance tf.float32)
            - network_postcondition | Type := output_class isinstance int (corresponding to label in image_label_set) 

        Returns: Propositional Logic Formula Given Relationship Between Variable States 
        """
        # if any adversarial attack state is true, then check for network's classification e.g. output state is it's output_class
        if (Adversarial.pgd_attack_state == True and RobustnessTrace.smt_satisfiability_state == True):
            return True
        
        elif (Adversarial.fgsm_attack_state == True and RobustnessTrace.fgsm_perturbation_correctness_state == True):
            return True
        
        elif (Adversarial.norm_perturbation_attack_state == True and RobustnessTrace.correctness_under_lp_perturbation_status == True):
            return True

        else:
            return False

    def propositional_satisfiability_formula(self):
        """Synthesize logical formula translated through encoding convolutional network as a constraint-satisfaction problem with respect to pre-condition and post-condition after network state-transition e.g. forwardpropagation. 
        
        Note, the implication is satisfied given the network_postcondition e.g. output_state given perturbation norm and input_image as network_precondition

        Note, according to Hoare logic with regards to propositional logic, the implementation of a function is partially correct with respect to its specification if, assuming the precondition is true just before the function executes, then if the function terminates, the postcondition is true

        Note that x := input_class and y := output_class

        Formally written as: Network ⊢ SAT ⟺ P(x,y) ⇒ Q(x,y) | P | ∀ x ∧ ∀ y     

        """

        raise NotImplementedError


class BoundedCryptoNetworkSolver(BoundedNetworkSolver):
    '''
        Compute Model Checker on Local Models in Federated Environment.
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''
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

    @staticmethod
    def verify_smt():
        '''
        Verify given the updated state of the Kripke node/world of output state given the variable state of 'RobustnessTrace.smt_satisfiability_state'

        '''
        raise NotImplementedError

    @staticmethod
    def verify_adversarial_example_not_created():
        adv = RobustnessTrace.adversarial_example_not_created_trace()
        if (adv == False):
            RobustnessTrace.adversarial_sample_created = False
            return False
        elif (adv == True):
            RobustnessTrace.adversarial_sample_created = True
            return True

    @staticmethod
    def verify_brightness_perturbation_robustness():
        '''Verify robustness given brightness perturbation norm-bounded attack against each input_image passed to network. '''
        raise NotImplementedError

    @staticmethod
    def verify_norm_perturbation_robustness():
        '''Verify robustness given l-norm (l^2, l-infinity, l-1) bounded perturbation attack against each input_image passed to network. '''
        raise NotImplementedError

    @staticmethod
    def verify_fgsm_attack_robustness():
        raise NotImplementedError

    @staticmethod
    def verify_pgd_attack_robustness():
        raise NotImplementedError
