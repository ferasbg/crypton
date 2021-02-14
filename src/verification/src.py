#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig

import os
import sys
import pickle
import random
import keras
import tensorflow

from verification.specification import Specification
from hyperproperties import SafetyProperties, RobustnessProperties, LivenessProperties


class Verification():
    """
        Description: Check each trace property defined in the formal specifications in `verification.specification`. Use solver to process bounded network state abstraction to compute satisfiability for each trace property.
        Args:
        Returns:
        Raises:
        References:
        Examples:
    """

    def __init__(self):
        '''
            Description: Store the formal specifications and each trace property. Also store the state for the parameters required for each verification technique.
            Args:
            Returns:
            Raises:
            References:
            Examples:
        '''
        self.specification = Specification()
        self.safety_properties = SafetyProperties()
        self.robustness_properties = RobustnessProperties()
        self.liveness_properties = LivenessProperties()

    @staticmethod
    def problem_formulation():
        '''
            Description: Define the state of the required arguments in order to solve the verification problem. Now this will depend on what specific specification we are checking for. Leave as problem_formulation, but then note the specification / property being checked.
            Args:
            Returns:
            Raises:
            References:
            Examples:
        '''
        raise NotImplementedError


class MPCSolver():
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
    raise NotImplementedError

if __name__ == '__main__':
    # create instance of required network state given specification
    Verification.problem_formulation()
    # iterate over each trace property: sequential workflow of property checking where `src.specification` stores the object state for each trace property whereas verification does the property check

    