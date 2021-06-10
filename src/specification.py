#!/usr/bin/env python3
# Copyright 2021 Feras Baig

import os
import sys
import sympy

import keras
import tensorflow as tf

from nn.network import Network

'''
- Extract succinct input-output characterizations of the network behavior, and store property inference algorithms for each property type.
- Given a classification deep neural network N with an input region Θ, the robustness property holds if and only if (<-->) all inputs within the input region Θ have the same label, i.e., ∀x [0] , y [0] ∈ Θ =⇒ ϱ(x^[0]) = ϱ(y^[0]).

% robustness trace property set τ:
    - robustness trace 1: given lp-norm perturbation, the euclidean distance of certified_accuracy (given proportion of correct classifications out of total classifications in training iteration) under certain threshold p to formally guarantee robustness of the network.
    - robustness trace 2: given projected gradient descent attack meant to distort backpropagation process, assert that model updates its convergence to local minima with gradient descent correctly given bounds
    - robustness trace 3: network is not making misclassifications with respect to L-norm (infinity, l^2, l-1)

- https://arxiv.org/pdf/1904.13215.pdf
- https://people.eecs.berkeley.edu/~sseshia/pubdir/atva18.pdf (3.2)

# just analyze how adversarial attacks affect convergence, and focus on nn optimization by having math to explain the phenomenon, e.g. confirming expectations or contradicting it with truth
# remove all these traces and focus on how they help contribute to the optimization formulation you have setup when relating an adversarial attack, the gradients of a client and server model, and its convergence
# all you did was fake progress and avoid the math involved in really making the goal of your project work which was impossible because you never scoped it down to specifics
# input-output robustness is deductively simple, e.g. self-confirmation over alchemy-like stupidity; focus on what you are measuring for
# stop twiddling ur thumbs and solve hard problems that are scoped...

Define the trace property given the constraint-satisfaction logical problem that the model checker is checking against. If the output state vector norm satisfies the specification, then the constraint and the model is certified to be robust given the SMT Solver.

In other words, a proportion of the reachable states satisfy the robustness threshold for accuracy with respect to n number of iterations (batches) and image samples.

Simply stated, we compare the pre-condition and relate it to the post-condition computed from keras.layers.Dense(10, activation='softmax') under the constraint of at least one norm perturbation or adversarial attack.

Relating to the idea of Kripke Structures for modeling state-transitions, we reduce our state space to evaluate by comparing the input and output layers and their pre and post conditions respectively in order to optimize computational efficiency and to maintain scalability and precision under a constraint of norm-bounded adversary for robustness evaluation.

If we would evaluate the hidden layers and their state-transition relationships with respect to the model's behavior to optimize its computations, we would need abstract interpretation that adheres to formal rigor when generating abstract conjugates of concrete layers and their node-to-node interactions, which is out of the focus of this project.

Best way to compute brightness perturbation norm is to iterate the function of maximizing brightness for each pixel for each input image.
    
Formal Notation: Trace ⊢ SAT ⟺ B(P(x,y)) ⇒ Q(x,y) | ∀ x ∧ ∀ y
    # this notation doesn't tell me anything other than confirming that an input helps converge a model after some defined steps

Create the adversarial attack, then perform adversarial analysis and check against trace property, that is stored here that defines the success metrics for each trace property given the state of the network given the adversarial attack.

Formal Notation: Given η=ϵ sign(∇ₓ J(θ,x,y)), for Trace ⊢ SAT ⟺ P(x,y) ⇒ Q(x,y) | x := η(x) | ∀ x ∧ ∀ y

robustness trace property 2: input-output relation comparing the perturbed image to a non-perturbed output given applied perturbation_epsilon to attack network with distortion to input_image.

Apply l_p perturbation norm for each input_image to maximize its loss.

Note, if the perturbations don't change the output label for pixelwise_perturbation_ɛ = 0.{1,2,3,4,5}, then there is no adversarial_example created, which satisfies the desired input-output relation between the perturbation_epsilon during data pre-processing. Other implementations apply the perturbation epsilon deeper into the network, but for maintaining dimensionality (and other reasons specified in the paper), the earlier the perturbations applied, the better.


'''


# variables to then use for trace property
adversarial_sample_created = False
robustness_threshold_state = False
counterexample_verificationState = False
lp_perturbation_status = False # l_p vector norm perturbation
correctness_under_lp_perturbation_status = False
brightness_perturbation_status = False
correctness_under_brightness_perturbation = False # formula to define vector norm by making every pixel "bright" 
fgsm_perturbation_attack_state = False
fgsm_perturbation_correctness_state = False
pgd_attack_state = False
pgd_correctness_state = False
