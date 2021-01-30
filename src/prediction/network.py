#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig

import logging
import os
import random
import time
import argparse

import keras
import numpy as np
import torch
from torch import nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

import helpers
from prediction.helpers import load_model

class Network(nn.Module):
    """
    Args:
        - pretrained_network: weights & architecture of VGG network
    Returns:
        Network (Type: nn.Module.Sequential)

    Raises:
        RaiseError: if model_layers not correctly appended and initialized (sanity check), if assert ObjectType = False

    References:
        - https://arxiv.org/abs/1409.1556

"""

    def __init__(self):
        super(Network, self).__init__()
        self.model = load_model()
        self.num_classes = 20
        self.batch_size = 64
        self.learning_rate = 1e-4
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
        self.requires_grad = True

        # specification trace variables
        self.reluUpperBound = 0 # upper bounds for each layer for symbolic interval
        self.reluLowerBound = 0 # lower bounds for each layer for symbolic interval analysis, based on state, specification is met / not met
        self.verificationState = False



    def create_model(self):
        """
        Build VGG-16 DCNN.

        Args:
            - num_classes: 20
            - pretrained_network: default=VGG16

        Raises:


        Returns:
            - type: Model Tensor Object

        References:
            - https://github.com/pochih/FCN-pytorch/blob/master/python/fcn.py

        """

        model = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

        return model

    

    def getSymbolicIntervalBounds(self):
        """Return computed network state and convert to symbolic abstractions + temporal signals for property inference"""
        pass


    def sendNetworkState(self):
        """Get network object state, via semantics and numerical representation. Deduce symbolic representation with `src.verification.symbolic_representation` in order to represent the constraints and network state."""
        pass


if __name__ == '__main__':
    Network()


