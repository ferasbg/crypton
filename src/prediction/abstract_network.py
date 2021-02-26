#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig

import os
import sys
import random
import time
import argparse
import sympy as sym

import numpy as np
from PIL import Image
import random
import tensorflow as tf
import keras

from prediction.network import Network
from keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPool2D, Softmax, UpSampling2D, ReLU, Flatten, Input, BatchNormalization, GaussianNoise, GaussianDropout

class AbstractNetwork(Network):
    '''
    Description: Finite-State Abstract Interpretation (A.I) for Computing Conditional Affine Transformations to Compute Abstract Domain Against Abstract Layers to Check Against Safety Trace Property Specifications.
    Args: tf.keras.Model
    Returns: AbstractNetwork
    Raises: BooleanError if lp_norm_perturbation_state=false e.g. input_image_set in perturbed_data_generator given perturbed_network_layer appended to network
    References: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8418593
    Examples:

    '''

    def build_abstract_conv_layer(self):
        raise NotImplementedError

    def build_abstract_max_pooling_layer(self):
        raise NotImplementedError

    def build_abstract_relu_layer(self):
        """ReLU to CAT."""
        raise NotImplementedError

    def build_abstract_domain(self):
        raise NotImplementedError

    def compute_abstract_domain_bounds(self):
        raise NotImplementedError

    def build_abstract_layers(self):
        raise NotImplementedError

    def build_zonotope_abstract_domain():
        raise NotImplementedError

    def relu_abstract_transformer():
        raise NotImplementedError

    def conv2d_abstract_transformer():
        raise NotImplementedError

    def dense_abstract_transformer():
        raise NotImplementedError

    def maxpool2d_abstract_transformer():
        raise NotImplementedError

    def get_greatest_robustness_bound():
        raise NotImplementedError

    def compute_reachable_states(network):
        if network.check_perturbation_layer() == True:
            # compute reachable states of perturbed network
            return false

        else:
            return false


