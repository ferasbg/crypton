#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig

import os
import sys
import random
import time
import argparse

import keras
import numpy as np
from PIL import Image
import random
import tensorflow as tf
from keras.applications.vgg16 import VGG16

from prediction.network import Network


class BoundedNetwork(Network):
    '''
    Description: Create a BoundedNetwork for BoundPropagation(BoundedNetwork) for BoundedNetwork is Finite-State State Abstraction
    Args: tf.keras.Model
    Returns: BoundedNetwork(tf.keras.Model)
    Raises: Error given NetworkState
    References: NULL
    Examples:

    '''
    raise NotImplementedError


class Formulate(BoundedNetwork):
    '''
    Description: Mathematical Formulation of L-p Norm Perturbation 
    Args:
    Returns:
    Raises:
    References:
    Examples:
    '''
    
    raise NotImplementedError


if __name__ == '__main__':
    raise NotImplementedError
