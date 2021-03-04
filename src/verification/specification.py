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
    robustness_threshold_state = False
    counterexample_verificationState = False
    lp_perturbation_status = False # l_p vector norm perturbation
    correctness_under_lp_perturbation_status = False
    brightness_perturbation_status = False
    correctness_under_brightness_perturbation = False # accuracy under brightness perturbation with (1-sigma) threshold
    fgsm_perturbation_attack_state = False
    fgsm_perturbation_correctness_state = False
    pgd_attack_state = False
    pgd_correctness_state = False
    smt_satisfiability_state = False # note that input_image is perturbed under some norm-bounded adversarial attack, but the input_class is constant as a precondition, append precondition of adversarial attack as well, this is what sets up the verification problem

    @staticmethod
    def adversarial_example_not_created_trace():
        '''
        
        '''
        
        # if the classified output class matches the correct output class
        if (RobustnessTrace.lp_perturbation_status == True and RobustnessTrace.getRobustnessThresholdState() == True): # when perturbation epsilon and gaussian noise vector applied to input_image before input is passed to ImageDataGenerator and keras.layers.Input
            return "Success! Model's accuracy under adversarial training exceeds the robustness threshold given the norm-bounded adversarial attack."
        else:
            return False

    @staticmethod
    def brightness_perturbation_norm_trace():
        """Best way to compute brightness perturbation norm is to iterate the function of maximizing brightness for each pixel for each input image.
        
        Formal Notation: Trace ⊢ SAT ⟺ B(P(x,y)) ⇒ Q(x,y) | ∀ x ∧ ∀ y
        """
        if (RobustnessTrace.brightness_perturbation_status == True and RobustnessTrace.correctness_under_brightness_perturbation == True):
            return "Brightness perturbation norm trace property checked out successfully."

        elif (RobustnessTrace.correctness_under_brightness_perturbation == False):
            return "Brightness perturbation norm trace property failed."
        

    @staticmethod
    def l_perturbation_norm_trace(perturbation_epsilon):
        '''
        robustness trace property 2: input-output relation comparing the perturbed image to a non-perturbed output given applied perturbation_epsilon to attack network with distortion to input_image.

        Apply l_p perturbation norm for each input_image to maximize its loss.

        Note, if the perturbations don't change the output label for pixelwise_perturbation_ɛ = 0.{1,2,3,4,5}, then there is no adversarial_example created, which satisfies the desired input-output relation between the perturbation_epsilon during data pre-processing. Other implementations apply the perturbation epsilon deeper into the network, but for maintaining dimensionality (and other reasons specified in the paper), the earlier the perturbations applied, the better.

        '''
        if (RobustnessTrace.lp_perturbation_status == True and RobustnessTrace.correctness_under_lp_perturbation_status == True):
            return "L-p norm perturbation trace successfully checked out."
        else:
            return "L-p norm perturbation trace failed. This neural network has successfully been affected in terms of adversarial example generation, and can lead to much disastrous faults if launched in production. Hotfix network architecture with training iterations."

    @staticmethod
    def pgd_attack_trace():
        """Create the adversarial attack, then perform adversarial analysis and check against trace property, that is stored here that defines the success metrics for each trace property given the state of the network given the adversarial attack."""
        if (RobustnessTrace.pgd_attack_state == True and RobustnessTrace.pgd_correctness_state == True):
            return "Success! Network is adversarially robust against PGD attacks."
        else:
            return "PGD attack trace failed. This neural network has successfully been affected in terms of adversarial example generation, and can lead to much disastrous faults if launched in production. Hotfix network architecture with training iterations."

    @staticmethod
    def fgsm_attack_trace():
        '''
        Formal Notation: Given η=ϵ sign(∇ₓ J(θ,x,y)), for Trace ⊢ SAT ⟺ P(x,y) ⇒ Q(x,y) | x := η(x) | ∀ x ∧ ∀ y
        
        '''
        if (RobustnessTrace.fgsm_perturbation_attack_state == True and RobustnessTrace.fgsm_perturbation_correctness_state == True):
            return "FGSM attack trace successfully checked out. Given the input variance of the fast gradient sign method with F(P(x,y)), the output state Q(x,y) was consistent under the robustness trace for FGSM."
        else:
            return "FGSM attack trace failed. This neural network has successfully been affected in terms of adversarial example generation, and can lead to much disastrous faults if launched in production. Hotfix network architecture with training iterations."

    @staticmethod
    def smt_constraint_satisfiability_trace():
        '''Define the trace property given the constraint-satisfaction logical problem that the model checker is checking against. If the output state vector norm satisfies the specification, then the constraint and the model is certified to be robust given the SMT Solver.
        
        In other words, a proportion of the reachable states satisfy the robustness threshold for accuracy with respect to n number of iterations (batches) and image samples.
        
        Simply stated, we compare the pre-condition and relate it to the post-condition computed from keras.layers.Dense(10, activation='softmax') under the constraint of at least one norm perturbation or adversarial attack.

        Relating to the idea of Kripke Structures for modeling state-transitions, we reduce our state space to evaluate by comparing the input and output layers and their pre and post conditions respectively in order to optimize computational efficiency and to maintain scalability and precision under a constraint of norm-bounded adversary for robustness evaluation.

        If we would evaluate the hidden layers and their state-transition relationships with respect to the model's behavior to optimize its computations, we would need abstract interpretation that adheres to formal rigor when generating abstract conjugates of concrete layers and their node-to-node interactions, which is out of the focus of this project.
        '''
        if (RobustnessTrace.smt_satisfiability_state == True):
            return "Success! The model checker has verified the postcondition of the output state of the convolutional network satisfies the constraint of the specification."

        else:
            return "The model checker has failed to verify the postcondition of the neural network's output state. Iterate on adversarial training and examples to make your model more robust, and try again."

    @staticmethod
    def getRobustnessThresholdState():
        return RobustnessTrace.robustness_threshold_state

if __name__ == '__main__':
    RobustnessTrace()
