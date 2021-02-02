#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig

import logging
import os
import random
import time
import argparse

import keras
import numpy as np
from PIL import Image
import random
import tensorflow as tf
from keras.applications.vgg16 import VGG16


from helpers import load_model

class Network():
    """
    Args:
        - pretrained_network: weights & architecture of VGG network
    Returns:

    Raises:
        RaiseError: if model_layers not correctly appended and initialized (sanity check), if assert ObjectType = False

    References:
        - https://arxiv.org/abs/1409.1556
        - https://github.com/pochih/FCN-pytorch/blob/master/python/fcn.py
        -
"""

    def __init__(self):
        super(Network, self).__init__()
        # labels
        self.num_classes = 20
        self.model = VGG16()



    def forward(self, input_tensor):
        pass

    def getSymbolicIntervalBounds(self):
        """Return computed network state and convert to symbolic abstractions + temporal signals for property inference"""
        pass


    def sendNetworkState(self):
        """Get network object state, via semantics and numerical representation. Deduce symbolic representation with `src.verification.symbolic_representation` in order to represent the constraints and network state."""
        return self.model


if __name__ == '__main__':
    pass

