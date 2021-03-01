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
from keras.datasets.cifar10 import load_data
import numpy as np


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
    x_train = []
    y_train = []
    x_test = []
    y_test = []

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
        # how many batches per epoch
        self.steps_per_epoch = 16

    def build_compile_model(self):
        # build layers of public neural network
        model = Sequential()
        # feature layers
        model.add(Conv2D(32, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPool2D((2, 2)))
        model.add(Flatten())
        # classification layers
        model.add(Dense(128, activation='relu',
                        kernel_initializer='he_uniform'))
        # 10 output classes possible
        model.add(Dense(10, activation='softmax'))
        model.add(Dense(units=2, activation='sigmoid'))
        # stochastic gd has momentum, optimizer doesn't use momentum for weight regularization
        optimizer = Adam(learning_rate=0.003)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer, metrics=['accuracy'])
        return model

    def train(self):
        # get cifar10-data first, and assign data and categorical labels as such
        data = Network.get_cifar_data()
        train_generator = ImageDataGenerator()
        train = train_generator.flow_from_directory(
            # note that x_train is directory to all train images
            directory=Network.x_train[:500], target_size=(32, 32), batch_size=32, class_mode='categorical')
        test_generator = ImageDataGenerator()
        test = test_generator.flow_from_directory(directory=Network.x_test[:500], target_size=(32, 32))

        self.model.fit(train, epochs=25, batch_size=32, validation_data=test)
        self.model.predict(self.model)
        print(self.model.summary())

    @staticmethod
    def get_cifar_data():

        # xy train is rgb image matrix and xy test is category labels for each respective image matrix
        '''
        x_test, y_test = load_batch(fpath)
        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 
        training images and 10000 test images.

        (50000, 32, 32, 3)
        (50000, 1)
        (10000, 32, 32, 3)
        (10000, 1)

        '''
        # x_train stores all of the train_images and y_train stores all the respective categories of each image, in the same order.
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        Network.x_train = x_train
        Network.y_train = y_train
        Network.x_test = x_test
        Network.y_test = y_test

    def evaluate_nominal(self):
        raise NotImplementedError

    @staticmethod
    def getClassificationState():
        return Network.classification_state


if __name__ == '__main__':
    # note that for each epoch_set we will iterate over each perturbation_epsilon and attack_type, defined in deploy.main
    graph = tf.compat.v1.get_default_graph()
    # instantiate tf.Session
    network = Network()
    print(network.model.summary())
