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
    Description: Finite-State Abstract Interpretation (A.I) for Computing Abstract Domain to Check Against Safety Trace Property Specifications.
    Args: tf.keras.Model
    Returns: AbstractNetwork
    Raises: BooleanError if lp_norm_perturbation_state=false e.g. input_image_set in perturbed_data_generator given perturbed_network_layer appended to network
    References: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8418593
    Examples:

    '''
    raise NotImplementedError


