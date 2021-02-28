#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig

import logging
import os
import random
import time
import argparse

import numpy as np
from PIL import Image
import random
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.models import Input, Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPool2D, Softmax, UpSampling2D, ReLU, Flatten, Input, BatchNormalization, GaussianNoise, GaussianDropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import numpy as np

class Network():
    '''
        Args: None

        Returns: tf.keras.Model

        Raises:
            RaiseError: if model_layers not correctly appended and initialized (sanity check), if assert ObjectType = False

        References:
            - https://arxiv.org/abs/1409.1556
            - https://github.com/pochih/FCN-pytorch/blob/master/python/fcn.py
    '''

    def __init__(self):
        '''
            Description:
            Args:
            Returns:
            Raises:
            References:
            Examples:
        '''

        # labels
        self.num_classes = 20
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
        self.shuffle = True # randomization
        self.weight_decay = 0.0005 # prevent vanishing gradient problem given diminishing weights over time
        self.momentum = 0.05 # gradient descent convergence optimizer

        # dimensions, and metadata for training
        self.width = 2048
        self.height = 1024


    def build_compile_model(self):
        # build layers of public network
        model = Sequential()


    def train(self):
        # setup training / test dataset and preprocessing
        # download the dataset for cifar10 directly, then partition dataset
        train_directory = './'
        test_directory = './'

        train_generator = ImageDataGenerator()
        train = train_generator.flow_from_directory(directory=train_directory, target_size=(224, 224))
        test_generator = ImageDataGenerator()
        test = test_generator.flow_from_directory(directory=test_directory, target_size=(224, 224))


    def freeze_feature_layers(self):
        '''
            Description:
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
    graph = tf.compat.v1.get_default_graph()


