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
        Description: Create keras model instance. Use build_and_compile_model() object function to append model layers and to initialize network for base/plaintext training.
        
        Args: None

        Returns: tf.keras.Model

        Raises:
            ValueError: if model_layers not correctly appended and initialized (sanity check), if assert ObjectType = False

        References:
            - https://arxiv.org/abs/1409.1556
    '''

    def __init__(self):

        # labels
        self.num_classes = 20
        self.epochs = 1000
        self.batch_size = 64 # maybe 128
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.kernel_size = 3
        self.stride = 2
        self.padding = 1
        self.dilation = 1
        self.epochs = 500
        self.input_channels = 3
        self.output_channels = 64 # number of channels produced by convolution
        self.image_size = [32, 32]
        self.bias = False
        self.weight_decay_regularization = 0.003 # stabilize convergence to local minima for gradient descent
        self.momentum = 0.05 # gradient descent convergence optimizer
        self.model = self.build_compile_model()
        self.dataset_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # how many batches per epoch
        self.steps_per_epoch = 16 

    def build_compile_model(self):
        # build layers of public neural network
        model = Sequential()
        # feature layers
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPool2D((2, 2)))
        model.add(Flatten())
        # classification layers
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        # 10 output classes possible
        model.add(Dense(10, activation='softmax')) 
        model.add(Dense(units=2, activation='sigmoid'))
        optimizer = Adam(learning_rate=0.003) # stochastic gd has momentum, optimizer doesn't use momentum for weight regularization
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def train(self, dataset):
        # append first 500 cifar-10 images to train_set and 500 next images to validation set
        train_set = {}
        validation_set = {}
        
        train_generator = ImageDataGenerator()
        train = train_generator.flow_from_directory(directory=train_set, target_size=(32, 32))
        validation_generator = ImageDataGenerator()
        validation = validation_generator.flow_from_directory(directory=validation_set, target_size=(32, 32))
        self.model.fit()

    def get_cifar_data(self):
        """
        Reference: https://github.com/exelban/tensorflow-cifar-10/blob/master/include/data.py
        
        """
        dataset = {}

        return dataset

    def freeze_feature_layers(self):
        for layer in self.model.layers:
            layer.trainable = False

    def evaluate_nominal(self):
        raise NotImplementedError


if __name__ == '__main__':
    # initialize tf.Session(sess) to initialize tf computational graph to track state-transition
    # note that for each epoch_set we will iterate over each perturbation_epsilon and attack_type, defined in deploy.main
    graph = tf.compat.v1.get_default_graph()
    network = Network()
    print(network.model.summary())    