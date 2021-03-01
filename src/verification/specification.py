#!/usr/bin/env python3
# Copyright 2021 Feras Baig

import os
import sys
import sympy

import keras
import tensorflow as tf

from nn.network import Network

'''
- Store hyperproperties e.g. robustness specifications to be checked given r ⊆ H. 
- Extract succinct input-output characterizations of the network behavior, and store property inference algorithms for each property type.
- Converge temporal specifications, and remove STL variations if not necessary.
'''

class RobustnessTrace():
    '''
        Description: Given a classification deep neural network N with an input region Θ, the robustness property holds if and only if (<-->) all inputs within the input region Θ have the same label, i.e., ∀x [0] , y [0] ∈ Θ =⇒ ϱ(x^[0]) = ϱ(y^[0]).

            % robustness trace property set τ:
                - robustness trace 1: given lp-norm perturbation, the euclidean distance of certified_accuracy (given proportion of correct classifications out of total classifications in training iteration) under certain threshold p to formally guarantee robustness of the network.
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
    lp_perturbation_status = False # l_p vector norm perturbation
    brightness_perturbation_status = False
    # accuracy under brightness perturbation with (1-sigma) threshold
    correctness_under_brightness_perturbation = False
    fgsm_perturbation_attack_state = False
    fgsm_perturbation_correctness_state = False
    pgd_attack_state = False
    pgd_correctness_state = False

    def robustness_bound_check(self):
        """robustness trace property 1: robustness_threshold and robustness_region given output vector norm. 
        
        Note, given the lp-norm perturbation, this trace property will be evaluated given the euclidean distance of certified_accuracy (given proportion of correct classifications out of total classifications in training iteration) under certain threshold p to formally guarantee robustness of the network.
        
        """
        # if output vector is inside robustness region, then specification is met
        raise NotImplementedError

    @staticmethod
    def adversarial_example_not_created():
        """
        robustness trace property 2: input-output relation comparing the perturbed image to a non-perturbed output given applied perturbation_epsilon to attack network with distortion to input_image.
        
        Note, if the perturbations don't change the output label for pixelwise_perturbation_ɛ = 0.{1,2,3,4,5}, then there is no adversarial_example created, which satisfies the desired input-output relation between the perturbation_epsilon during data pre-processing. Other implementations apply the perturbation epsilon deeper into the network, but for maintaining dimensionality (and other reasons specified in the paper), the earlier the perturbations applied, the better.
        """
        # use when perturbation_status=True
        if (RobustnessTrace.lp_perturbation_status == True): # when perturbation epsilon and gaussian noise vector applied to input_image before input is passed to ImageDataGenerator and keras.layers.Input
            # if the classified output class matches the correct output class
            if (Network.getClassificationState() == True):
                return True
        
        return False

    @staticmethod
    def brightness_perturbation_norm_trace():
        """Best way to compute brightness perturbation norm is to iterate the function of maximizing brightness for each pixel for each input image. Define adversarial operators in Adversarial, and write trace properties and assert checks here."""
        if (RobustnessTrace.brightness_perturbation_status == True and RobustnessTrace.correctness_under_brightness_perturbation == True):
            return "Brightness perturbation norm trace property checked out successfully."

        elif (RobustnessTrace.correctness_under_brightness_perturbation == False):
            return "Brightness perturbation norm trace property failed."
        

    @staticmethod
    def l_perturbation_norm_trace(self, norm_perturbation_correctnessState):
        '''Apply l_p perturbation norm for each input_image to maximize its loss.'''
        if (RobustnessTrace.lp_perturbation_status == True and norm_perturbation_correctnessState == True):
            return "L-p norm perturbation trace successfully checked out."
        else:
            return "L-p norm perturbation trace failed."

    @staticmethod
    def pgd_attack_trace(self):
        """Create the adversarial attack, then perform adversarial analysis and check against trace property, that is stored here that defines the success metrics for each trace property given the state of the network given the adversarial attack."""
        raise NotImplementedError

    @staticmethod
    def fgsm_attack_trace(self):
        if (RobustnessTrace.fgsm_perturbation_attack_state == True and RobustnessTrace.fgsm_perturbation_correctness_state == True):
            return "FGSM attack trace successfully checked out. Given the input variance of the fast gradient sign method with F(P(x,y)), the output state Q(x,y) was consistent under the robustness trace for FGSM."
        else:
            return "FGSM attack trace failed. This neural network has successfully been affected in terms of adversarial example generation, and can lead to much disastrous faults if launched in production. Hotfix network architecture with training iterations."

    @staticmethod
    def smt_solver_trace_constraint(self):
        '''Define the trace property given the constraint-satisfaction logical problem that the model checker is checking against. If the output state vector norm satisfies the specification, then the constraint and the model is certified to be robust given the SMT Solver.'''
        raise NotImplementedError

    @staticmethod
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



