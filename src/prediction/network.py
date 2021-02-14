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


class Network():
    """
    Args: None

    Returns: tf.keras.Model

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
        self.epochs = 1000
        self.batch_size = 64 # maybe 128
        self.learning_rate = 1e-4
        self.batch_size = 64
        self.kernel_size = 3
        self.stride = 2
        self.padding = 1
        self.dilation = 1
        self.epochs = 500
        self.input_channels = 3
        self.output_channels = 64 # number of channels produced by convolution
        self.image_size = [224, 224]
        self.bias = False
        self.shuffle = True
        self.weight_decay = 0.0005
        self.momentum = 0.99

        # dimensions, and metadata for training
        self.width = 2048
        self.height = 1024
        # self.optimizer, self.schedule

        # specification trace variables
        self.reluUpperBound = 0 # upper bounds for each layer for symbolic interval
        self.reluLowerBound = 0 # lower bounds for each layer for symbolic interval analysis, based on state, specification is met / not met
        self.verificationState = False

        # iou
        self.mean_iou = 0
        self.true_positive_pixels = 0
        self.false_positive_pixels = 0
        self.false_negative_pixels = 0
        self.misclassified_pixels = 0
        self.frequency_weighted_iou = 0


        # acc
        self.mean_acc = 0
        self.pixelwise_acc = 0

        # perturbation
        self.gaussian_noise_factor = 0.10
        self.perturbation_factor = 0.05


    def freeze_feature_layers(self):
        raise NotImplementedError

    def forward(self, input_tensor):
        raise NotImplementedError

    @property
    def getSymbolicIntervalBounds(self):
        """Return computed network state and convert to symbolic abstractions + temporal signals for property inference"""
        raise NotImplementedError

    @property
    def sendNetworkState(self):
        '''
            Description: Get network object state, via semantics and numerical representation. Deduce symbolic representation with `src.verification.symbolic_representation` in order to represent the constraints and network state.
            Args:
            Returns:
            Raises:
            References:
            Examples:
        '''

        raise NotImplementedError

    def evaluate_nominal(self):

        '''
            Description:
            Args:
            Returns:
            Raises:
            References:
            Examples:
        '''

        raise NotImplementedError


if __name__ == '__main__':
    # initialize tf.Session(sess) to initialize tf computational graph to track state-transition
    graph = tf.get_default_graph()
    saver = tf.train.Saver()


