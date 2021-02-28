#!/usr/bin/env python3
# Copyright 2021 Feras Baig

import os
import sys

import keras
import tensorflow as tf

from prediction.network import Network
from hyperproperties import RobustnessProperties, SafetyProperties

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


class CheckTraceData():
    '''
        Description: Check if all of the robustness specification formalisms are correct.
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''
    raise NotImplementedError


if __name__ == '__main__':
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



