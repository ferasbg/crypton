#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig

import argparse
import collections
import logging
import os
import pickle
import random
import sys
import time
import warnings

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import tqdm
from keras import backend as K
from keras import optimizers, regularizers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.datasets import cifar10
from keras.datasets.cifar10 import load_data
from keras.layers import (Activation, BatchNormalization, Conv2D,
                          Conv2DTranspose, Dense, Dropout, Flatten,
                          GaussianDropout, GaussianNoise, Input, MaxPool2D,
                          ReLU, Softmax, UpSampling2D)
from keras.layers.core import Lambda
from keras.models import Input, Model, Sequential, load_model, save_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow import keras

# The goal here is to certify robustness of adversarial examples and the models at the client-level and at the server-side model level (trusted aggregator, parameter evaluation)
# the relationship between generating robust adversarial examples and developing adversarially robust models is that both conditions help each other
# reference: https://people.eecs.berkeley.edu/~sseshia/pubdir/vnn19.pdf

'''
Mathematical Formulation of Adversarially Robust Federated Optimization With Neural Structured Learning:

Variables:
  - neural network function (map to an output probability with a softmax)
  - sampling algorithm for clients
  - adaptive federated optimization algorithm given adversarially regularized client models
  - neural structured learning algorithm given neural network (formalize feature decomposition, regularization --> federated strategy --> server-side parameter evaluation-under-attack (robustness))
  - server-side parameter evaluation algorithm in federated setting
  - checking server-side model (trusted aggregator) against server-side specifications for certification of adversarial robustness (both in terms of the examples and the accuracy under perturbation attacks --> extrapolate robustness via federated accuracy-under-attack to abstraction given decision formulation in decision_formulations (set of all specifications/checks))

Statements:
  -
  -
  -

Each statement will test the system at the client and server-side level in terms of the algorithms used between the two components as well.


'''

class Specification(object):

  '''
      robustness trace property set τ:
          - robustness trace 1: given lp-norm perturbation, the euclidean distance of certified_accuracy (given proportion of correct classifications out of total classifications in training iteration) under certain threshold p to formally guarantee robustness of the network.
          - robustness trace 3: network is not making misclassifications with respect to L-norm (infinity, l^2, l-1)
          - robustness trace 4: input-output relation comparing the perturbed image to a non-perturbed output given applied perturbation_epsilon to attack network with distortion to input_image.
          - robust kl-divergence: kl-divergence for server-side federated accuracy under perturbation attack
      - note: Given a classification deep neural network N with an input region Θ, the robustness property holds if and only if (<-->) all inputs within the input region Θ have the same label, i.e., ∀x [0] , y [0] ∈ Θ =⇒ ϱ(x^[0]) = ϱ(y^[0]). : fancy way of checking if the property that wants an accurate label holds and robustness itself translates to a threshold given the batch_episode set per epoch or a set of epochs iterating over the defined # of rounds.
      - objective: just analyze how adversarial attacks affect convergence, and focus on nn optimization by having math to explain the phenomenon, e.g. confirming expectations or contradicting it with truth
      - note: If the perturbations don't change the output label for pixelwise_perturbation_ɛ = 0.{1,2,3,4,5}, then there is no adversarial_example created, which satisfies the desired input-output relation between the perturbation_epsilon during data pre-processing.

      References:
          - https://arxiv.org/pdf/1904.13215.pdf
          - https://people.eecs.berkeley.edu/~sseshia/pubdir/atva18.pdf (3.2)

      - misc: smoothing, min-max perturbation, loss maximization as contradiction

  '''

  def compute_admissibility_constraint(x):
    '''
    The Admissibility Constraint (1) ensures that the adversarial input x∗ belongs to the space of admissible perturbed inputs.

    Eq: x∗ ∈ X˜

    How do you know what's an "admissible" input? An input that maximizes loss? Remember how optimization desires for the opposite? Formulations here are contradicting each other, but these are general statements not included in any of the functions in this file.


    '''

    return x

  def compute_distance_constraint(adv_step_size, adv_grad_norm):
    '''
    The Distance Constraint (2) constrains x∗ to be no more distant from x than α.

    Eq: D(µ(x, x∗), α)

    '''
    x = []
    return x

  def compute_target_behavior_constraint(a, x, x_prime, beta):
    '''
    The Target Behavior Constraint (3) captures the target behavior of the adversary as a predicate A(x, x∗, β) which is true if the adversary changes the behavior of the ML model by at least β modifying x to x∗. If the three constraints hold, then we say that
    the ML model has failed for input x. We note that this is a so called “local” robustness property for a specific input x, as
    opposed to other notions of “global” robustness to changes to a population of inputs (see Dreossi et al. [2018b]; Seshia et al. [2018].

    Eq: A(x, x∗, β)

    '''
    x = []
    return x


