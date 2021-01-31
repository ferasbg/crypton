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
from torch.autograd import Variable
import random


from helpers import load_model

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
        - https://github.com/pochih/FCN-pytorch/blob/master/python/fcn.py
        -
"""

    def __init__(self, pretrained_network):
        super(Network, self).__init__()
        # network
        self.model = nn.Sequential(
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
        # labels
        self.num_classes = 20


    def forward(self, input_tensor):
        pass

    def getSymbolicIntervalBounds(self):
        """Return computed network state and convert to symbolic abstractions + temporal signals for property inference"""
        pass


    def sendNetworkState(self):
        """Get network object state, via semantics and numerical representation. Deduce symbolic representation with `src.verification.symbolic_representation` in order to represent the constraints and network state."""
        return self.model


if __name__ == '__main__':
    # weight binaries
    pretrained_network = load_model()
    network = Network(pretrained_network)
    print(network)
