#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig

import logging
import os
import random
import time

import keras
import numpy as np
import torch
from torch import nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

import helpers

class Network(nn.Module):
    """
    Args:
        Network:
        - self.batch_size: number of images per operation given source input image
        - self.num_classes: number of labels given segmentation operation
        - self.image_size = [224,224]: default for torch.nn.Module and VGG-16
        - self.upper_bound: upper bounds for each layer for symbolic interval
        - self.lower_bound: lower bounds for each layer for symbolic interval analysis, based on state, specification is met / not met
        - self.mask_size: size for segmentation mask or kernel size
        - self.input_channels = 3: number of channels in input image
        - self.output_channels: number of channels produced by convolution
        - self.kernel_size = 3: size of convolving kernel
        - self.stride = []: stride of convolution, for default = 2
        - self.padding = []: set padding to 1
        - self.bias: bias constant for each computation during foward-pass traversal of the conv network
        - self.num_training: setup sample size for training batch set
        - self.num_validation
        - self.train_generator: keras-native function to train model
        - self.validation_generator: keras-native function to run concurrent process of validation (of classification tasks)

    Returns:
        Network (Type: nn.Module.Sequential)

    Raises:
        RaiseError: if model_layers not correctly appended and initialized

    References:
        - https://arxiv.org/abs/1409.1556
        - https://towardsdatascience.com/convolutional-neural-networks-mathematics-1beb3e6447c0
        - https://github.com/donnyyou/torchcv/tree/master/model/seg
        - https://towardsdatascience.com/gentle-dive-into-math-behind-convolutional-neural-networks-79a07dd44cf9
        - https://imgaug.readthedocs.io/en/latest/source/examples_segmentation_maps.html


"""

    def __init__(self,
        batch_size = 64,
        learning_rate = 0.003,
        kernel_size = 3,
        stride = 2,
        padding = 1,
        epochs = 1000,
        input_channels = 3,
        num_classes = 25,
        output_channels = 64,
        image_size = [224, 224],
        bias = False,
        shuffle = True
        ):
        super(Network, self).__init__()

        self.gaussian_noise_factor = 0.10
        self.perturbation_factor = 0.05
        self.reluUpperBound = 0
        self.reluLowerBound = 0



    def create_model(self):
        """

        Mask R-CNN for Instance Segmentation.

        Args:
            - input for layer i-1: Tensor

        Raises:


        Returns:
            - type: Dict[Tensor]: tensor of losses for each timestep

        References:
            - https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py


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

        for param in model.parameters():
            param.requires_grad = False




        return model


    def getSymbolicIntervalBounds(self):
        """Pass data to BoundedReLU Object"""

        pass


    def sendNetworkState(self):
        """Get network object state, via semantics and numerical representation. Deduce symbolic representation with `src.verification.symbolic_representation` in order to represent the constraints and network state."""
        pass






if __name__ == '__main__':
    # setup argparse
    # setup logger (during training)
    Network()
    Network().create_model()
    print(Network)


