#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig

import argparse
import logging
import os
import random
import time

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import optimizers, regularizers
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.datasets.cifar10 import load_data
from keras.layers import (Activation, BatchNormalization, Conv2D,
                          Conv2DTranspose, Dense, Dropout, Flatten,
                          GaussianDropout, GaussianNoise, Input, MaxPool2D,
                          ReLU, Softmax, UpSampling2D)
from keras.layers.core import Lambda
from keras.models import Input, Model, Sequential, load_model, save_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow import keras
from tensorflow.python.keras.applications.vgg16 import preprocess_input


class Network():
    '''
        Description: Create keras model instance. Use build_and_compile_model() object function to append model layers and to initialize network for base/plaintext training.

        Args: None

        Returns: keras.Model

        Raises:
            ValueError: if model_layers not correctly appended and initialized (sanity check), if assert ObjectType = False

        References:
            - https://arxiv.org/abs/1409.1556
    '''
    classification_state = False
    dataset_labels = ['airplane', 'automobile', 'bird',
                      'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # list of ints that represent the labels given self.dataset_labels so basically index the list of dataset_labels given y_train[i] with Network.dataset_labels[i]

    def __init__(self):

        # labels
        self.num_classes = 20
        self.epochs = 1000
        self.batch_size = 64  # maybe 128
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.kernel_size = 3
        self.stride = 2
        self.padding = 1
        self.dilation = 1
        self.epochs = 500
        self.input_channels = 3
        self.output_channels = 64  # number of channels produced by convolution
        self.image_size = [32, 32]
        self.bias = False
        # stabilize convergence to local minima for gradient descent
        self.weight_decay_regularization = 0.003
        self.momentum = 0.05  # gradient descent convergence optimizer
        self.model = self.build_compile_model()

    def build_compile_model(self):
        # build layers of public neural network
        model = Sequential()
        # feature layers
        model.add(Conv2D(32, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPool2D((2, 2)))
        model.add(Flatten())
        # classification layers
        model.add(Dense(128, activation='relu',
                        kernel_initializer='he_uniform'))
        # 10 output classes possible
        model.add(Dense(10, activation='softmax'))
         # stochastic gd has momentum, optimizer doesn't use momentum for weight regularization
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer, metrics=['accuracy'])
        return model

    def train(self):
        '''
        Train network on nominal multi-label classification problem. Make sure to allocate images for each test variation e.g. mpc_network, certified_mpc_network, certified_nominal_network

        x_test, y_test = load_batch(fpath)
        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        # partitioning dataset for different tests.
        x_val = x_train[-10000:]
        y_val = y_train[-10000:]
        x_train = x_train[:-10000]
        y_train = y_train[:-10000]

        Add dataset, and dataset_labels as parameter (overload this method or make re-write)

        '''
        # x_train stores all of the train_images and y_train stores all the respective categories of each image, in the same order.
        # get cifar10-data first, and assign data and categorical labels as such
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        x_train = x_train.reshape((-1, 32, 32, 3))
        x_test = x_test.reshape((-1, 32, 32, 3))

        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        generator = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False)

        history = self.model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))
        self.model.evaluate(x=x_test, y=y_test, verbose=0)

        self.model.save_weights('network.h5')
        print(history)
        

    @staticmethod
    def getClassificationState():
        return Network.classification_state


if __name__ == '__main__':
    # note that for each epoch_set we will iterate over each perturbation_epsilon and attack_type, defined in deploy.main
    graph = tf.compat.v1.get_default_graph()
    # instantiate tf.Session
    network = Network()
    print(network.model.summary())
    network.build_compile_model()
    network.train()