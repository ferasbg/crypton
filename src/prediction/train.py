import logging
import os
import random
import time

import keras
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from torch import nn
from torchvision import transforms


from network import Network
from helpers import load_model



class Train(Network):
    """
    Verifiably Robust Training And Dev(Train)/Validation(Test) for `Public` Neural Network. Compute nominal training and variant of verifiable computations with specifications during model training as well.

    Args: None

    Raises:
        -
        -


    Returns:
        -
        -
        -

    References:
        -
        -
        -



    """

    def __init__(self):
        super(Network, self).__init__()
        self.model = load_model()
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
        self.criterion = nn.BCEWithLogitsLoss()
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



    def forward(self, input_tensor):
            """Forward pass of DCNN with input_tensor which stores the image"""
            pass

    def evaluate(self):
        """Compute segmentation metrics."""
        pass


    def freeze_feature_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    Train()
    Train().freeze_feature_layers()
    print(Train())





