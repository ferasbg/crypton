#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig

import argparse
import os
import random
import sys
import time

import keras
import numpy as np
import sympy as sym
import tensorflow as tf
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Dense,
                          Flatten, GaussianDropout, GaussianNoise, Input,
                          MaxPool2D, ReLU, Softmax, UpSampling2D)
from PIL import Image

from prediction.network import Network


class AbstractNetwork(Network):
    '''
    Description: Finite-State Abstract Interpretation (A.I) for Computing Conditional Affine Transformations to Compute Abstract Domain Against Abstract Layers to Check Against Safety Trace Property Specifications.
    Args: tf.keras.Model
    Returns: AbstractNetwork
    Raises: BooleanError if lp_norm_perturbation_state=false e.g. input_image_set in perturbed_data_generator given perturbed_network_layer appended to network
    References: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8418593
    Examples:

    '''

    @staticmethod
    def build_abstract_conv_layer(self):
        raise NotImplementedError

    @staticmethod
    def build_abstract_max_pooling_layer(self):
        raise NotImplementedError

    @staticmethod
    def build_abstract_relu_layer(self):
        """ReLU to CAT."""
        raise NotImplementedError

    @staticmethod
    def build_abstract_domain():
        raise NotImplementedError

    @staticmethod
    def compute_abstract_domain_bounds():
        raise NotImplementedError

    @staticmethod
    def build_abstract_layers():
        raise NotImplementedError

    @staticmethod
    def build_zonotope_abstract_domain():
        raise NotImplementedError

    @staticmethod
    def relu_abstract_transformer():
        raise NotImplementedError

    @staticmethod
    def conv2d_abstract_transformer():
        raise NotImplementedError

    @staticmethod
    def dense_abstract_transformer():
        raise NotImplementedError

    @staticmethod
    def maxpool2d_abstract_transformer():
        raise NotImplementedError

    @staticmethod
    def get_greatest_robustness_bound():
        """Make sure to compare the robustness bounds and polytope robustness regions e.g. precision given polytope domain type e.g. zonotope."""
        raise NotImplementedError

    @staticmethod
    def compute_reachable_states(network):
        if isinstance(network, Network):
            # if there has been perturbations applied to the network, compute reachable states of perturbed network
            raise NotImplementedError
        raise NotImplementedError   

    @staticmethod

    def get_abstract_loss():
        raise NotImplementedError

    @staticmethod
    def create_adversarial_polytope():
        """Create bounded set representation via geometric figure after perturbations are applied OR we are generating finite representation of possible perturbations given perturbation_epsilon. Refer to ERAN."""
        raise NotImplementedError

    @staticmethod
    def meet_abstract_domain_operator():
        """The meet ( ) operator is an abstract transformer for set intersection: for an inequality expression E from Fig. 3, γ n (a) ∩ {x ∈ R n | x |= E} ⊆ γ n (a E)."""
        raise NotImplementedError

    @staticmethod
    def join_abstract_domain_operator():
        """The join ( ) operator is an abstract transformer for set union: γ n (a 1 ) ∪ γ n (a 2 ) ⊆ γ n (a 1 a 2 )."""
        raise NotImplementedError

    @staticmethod
    def affine_transformer():
        """"""
        raise NotImplementedError

    @staticmethod
    def setup_robustness_bound():
        raise NotImplementedError

    @staticmethod
    def setup_robustness_property():
        raise NotImplementedError

    @staticmethod
    def create_abstract_domain():
        raise NotImplementedError
