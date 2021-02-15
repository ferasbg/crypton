#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig
import logging
import os
import random
import time
import argparse

import keras
import numpy as np
from PIL import Image
import random
import tensorflow as tf
from keras.applications.vgg16 import VGG16

from prediction.network import Network
from bound_propagation import BoundPropagation, Bounds
from stl import STLSpecification, Trace, STLCheckTraceData, STLSafetyTrace, STLRobustnessTrace


class HyperProperties():
    '''
        Description: Store hyperproperties to compute specifications in verification node, checking object state against violations of set of trace properties for H = {s, r, l} for s ⊆ H, r ⊆ H, l ⊆ H. Extract succinct input-output characterizations of the network behaviour, and store property inference algorithms for each property type.
        
        Args: None
        
        Raises: AssertTypeError
        
        Returns:
       
        References:
            - https://arxiv.org/pdf/1904.13215.pdf
            - https://people.eecs.berkeley.edu/~sseshia/pubdir/atva18.pdf (3.2)

    '''

    def __init__(self):
        self._robustness_properties = RobustnessProperties()
        self._safety_properties = SafetyProperties()


class RobustnessProperties(HyperProperties):
    '''
        Description: 
            - Store specifications to compute violations of robustness properties, which will be initialized to empty values.

            "Decision Formulation of Local Robustness: The decision version of this optimization problem states that, given a bound β and input x, the adversarial analysis problem is to find a perturbation δ such that the following formula is satisfied:
                [µ(δ) < β∧δ ∈ ∆] ⇒ [ fw(x+δ) 6∈ T(x)].
            "

            Global Robustness: One can generalize the previous notion of robustness by universally quantifying over all inputs x, to get the following formula, for a fixed β
                ∀x. ∀δ. ¬ϕ(δ)

            % robustness trace property set τ:
                - robustness trace 1: given lp-norm perturbation, the misclassified pixels is under certain threshold p 
                - robustness trace 2: given projected gradient descent attack meant to distort backpropagation process, assert that model updates its convergence to local minima with gradient descent correctly given bounds
                - robustness trace 3: network is not making misclassifications with respect to L-norm and bound propagation


        Args:
            - self.robustness_value = []: store robustness value
            - self.robustness_sensitivity: store robustness sensitivity
            - self.misclassified_pixels: number of misclassified pixels
            - self.correct_pixels: number of correctly classified pixels
            - self.unknown_pixels: number of unknown pixels without label
            - self.adversarial_pixels: number of perturbed pixels
            - self.numPixels = [224, 224]

        Raises:

        Returns:

        References:
    '''

    def __init__(self):
        super(RobustnessProperties, self).__init__()
        self.robustness_value = 0 # store robustness value
        self.robustness_sensitivity = 1 # store robustness sensitivity
        self.misclassified_pixels  = 0 # number of misclassified pixels
        self.correct_pixels = 0 # number of correctly classified pixels
        self.unknown_pixels = 0 # number of unknown pixels without label
        self.adversarial_pixels: 0 # number of perturbed pixels (not gaussian noise)
        self.numPixels = [224, 224] # image_size
        self.computation_time = 0 # store compute time for robustness verification, e.g. time/space, memory_usage, gpu_parallelization_ability=True, contract_formulationStatus=True, tightBounds=True 
        self.counterexample_verificationState = False

class SafetyProperties(HyperProperties):
    '''
        Description: Store specifications to compute violations of safety properties, which will be initialized to empty values. Define safety properties meant to assert network object state that adheres to safety constraints i.e. nothing bad is happening.
        Args:
            - safety trace 1: observance of network object state at timestep n when network is doing x
            - safety trace 2: model is negatively rewarding gaussian noise distribution (adding noise to pixels vs. perturbing kernels), a function of the preprocessor for the input image
            - safety trace 3: model is not affected by perturbed pixels given correct label annotation for each pixel, if this is incorrect, then safety is threatened because region of space may be incorrectly classified given individual pixels
            - safety trace 4: the model is violating less than x number of classifications given each pixel given the input frame matrix
        
        Returns:

        Raises:
            Error if iter in layers [instanceof = False], def checks(): ([for v in initialize_verification]: BoundPropagationTypeDefinitionError , PublicSymbolicIntervalAnalysisError) 

        References:

    '''

    def __init__(self):
        super(SafetyProperties, self).__init__()
        # define member variables that can apply to each safety property specification
        self.safetyPropertyState = False

if __name__ == '__main__':
    # instantiate objects that store computations and hyper>trace properties
    SafetyProperties()
    RobustnessProperties()
