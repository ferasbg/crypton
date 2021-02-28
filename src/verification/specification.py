#!/usr/bin/env python3
# Copyright 2021 Feras Baig

import os
import sys

import keras
import tensorflow as tf

from nn.network import Network

'''
- Store hyperproperties e.g. safety and robustness specifications to be checked given s ⊆ H, r ⊆ H. 
- Extract succinct input-output characterizations of the network behavior, and store property inference algorithms for each property type.
- Converge temporal specifications, and remove STL variations if not necessary.
'''

class RobustnessTrace():
    '''
        Description: Given a classification deep neural network N with an input region Θ, the robustness property holds if and only if (<-->) all inputs within the input region Θ have the same label, i.e., ∀x [0] , y [0] ∈ Θ =⇒ ϱ(x^[0]) = ϱ(y^[0]).

            % robustness trace property set τ:
                - robustness trace 1: given lp-norm perturbation, the misclassified pixels is under certain threshold p
                - robustness trace 2: given projected gradient descent attack meant to distort backpropagation process, assert that model updates its convergence to local minima with gradient descent correctly given bounds
                - robustness trace 3: network is not making misclassifications with respect to L-norm (infinity, l^2, l-1)

        Args:
        Returns:
        Raises:
        References:
            - https://arxiv.org/pdf/1904.13215.pdf
            - https://people.eecs.berkeley.edu/~sseshia/pubdir/atva18.pdf (3.2)
        Examples:
    '''
    # variables to then use for trace property
    adversarial_sample_created = False
    counterexample_verificationState = False
    robustness_sensitivity = 1 # store robustness sensitivity of perturbation norm e.g. possibly euclidean distance, given tuple of perturbation_norms to iteratively use

    def robustness_bound_check(self):
        # if output vector is inside robustness region, then specification is met
        raise NotImplementedError

    @staticmethod
    def adversarial_example_not_created():
        # return : bool
        # if the perturbations don't change the output label for pixelwise_perturbation_ɛ = 0.03, then there is no adversarial_sample (evaluate given dict_string of image_label for cifar-10)
        raise NotImplementedError

    def brightness_perturbation_norm_trace(self):
        raise NotImplementedError

    def l_perturbation_norm_trace(self):
        raise NotImplementedError
    
    def pgd_attack_trace(self):
        raise NotImplementedError

    def fgsm_attack_trace(self):
        raise NotImplementedError

    def smt_solver_trace_constraint(self):
        raise NotImplementedError

    def assert_mpc_reliability(self):
        raise NotImplementedError


class CheckTraceData():
    '''
        Description: Check if all of the robustness specification formalisms are correct. This is to circumvent any formalism faults given the formal statements for each robustness trace property.
        Args:
        Returns:
        Raises:
        References:
        Examples:
    '''
    @staticmethod
    def check_all_robustness_properties():
        raise NotImplementedError


if __name__ == '__main__':
    adv = RobustnessTrace.adversarial_example_not_created()
    if (adv == False):
        adversarial_sample_created = False
    elif (adv == True):
        adversarial_sample_created = True



