#!/usr/bin/env python3
# Copyright 2021 Feras Baig

import os
import sys

import keras
import tensorflow as tf

from prediction.network import Network
from hyperproperties import RobustnessProperties, SafetyProperties
from bound_propagation import BoundPropagation

class Specification():
    """
        Description: Core Formal Specifications for Deep Convolutional Neural Network. Note that the SafetyProperties & RobustnessProperties are parent classes that store a general-purpose set of property definitions, but the functions of the trace types will define the formal logic and specific trace itself, appended to its inherited property definition. Also, write and aggregate all specifications and sub-nodes in verification node to compute on network during its training and testing
        Args:
            self.bound_propagation = BoundPropagation(): store algorithm for bound propagation
            self._robustness_properties = RobustnessProperties(): store initialized robustness properties and compute specifications for robustness verification
            self._safety_properties = SafetyProperties(): : store initialized safety properties and compute specifications for safety verification

        Returns:
        Type: Specification ( object that stores metrics for all computations for safety verification, robustness, and liveness properties)

        Raises:
        References:
    """

    def __init__(self):
        # general property definitions
        self._robustness_properties = RobustnessProperties()
        self._safety_properties = SafetyProperties()
        # property check state
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
        Description: Specification of safety trace property that nests adversarial robustness and privacy trace properties.
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
            Definition 2 (Robustness): Given a classification deep neural network N with an input region Θ, the robustness property holds if and only if all inputs within the input region Θ have the same label, i.e., ∀x [0] , y [0] ∈ Θ =⇒ ϱ(x^[0]) = ϱ(y^[0]).
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''
    affine_output_vector = False
    adversarial_sample_created = False

    @staticmethod
    def affine_abstract_output_vector_state():
        # return : bool
        # if vector is inside robustness region, then specification is met
        raise NotImplementedError

    @staticmethod
    def adversarial_example_not_created():
        # return : bool
        # if the perturbations don't change the output label for pixelwise_perturbation_ɛ = 0.03, then there is no adversarial_sample (evaluate given dict_string of image_label for cifar-10)
        raise NotImplementedError




if __name__ == '__main__':
    Specification()
    t_f = RobustnessTrace.affine_abstract_output_vector_state()
    adv = RobustnessTrace.adversarial_example_not_created()
    if (t_f == True and adv == False):
        affine_output_vector = True
        adversarial_sample_created = False
    elif (t_f == True):
        affine_output_vector = True
    elif (adv == False):
        adversarial_sample_created = False
    elif (adv == True):
        adversarial_sample_created = True
    elif (t_f == False):
        affine_output_vector = False



