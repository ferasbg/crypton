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
from keras.datasets import cifar100
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
from keras.applications.vgg16 import preprocess_input
import tensorflow_datasets as tfds

class Network():
    '''
        Description: 
            - Create keras model instance. Use build_and_compile_model() object function to append model layers and to initialize network for base/plaintext training.
            - Note that a convolutional neural network is generally defined by a function F(x, θ) = Y which takes an input (x) and returns a probability vector (Y = [y1, · · · , ym] s.t. P i yi = 1) representing the probability of the input belonging to each of the m classes. The input is assigned to the class with maximum probability (Rajabi et. al, 2021).
        
        Args: None
        
        Returns: keras.models.Model
        
        Raises:
            ValueError: mismatch of model layer stack
        
        References:
            - https://arxiv.org/abs/1409.1556
    '''

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
        # self.model_grads = (k.gradients(self.model.layers[0].output, self.model.trainable_weights[0])) # get Conv2d grads to perturb the network for PGD

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
        # use sparse categorical cross entropy since each image corresponds to one label given only 1 scalar node valid given output one-hot vector in output layer
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer=optimizer, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        return model

    def build_uncompiled_model(self):
        # building uncompiled plaintext model
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
        return model

    def train_model(self):
        '''
        x_test, y_test = load_batch(fpath)
        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        partitioning dataset for different tests.

        x_val = x_train[-10000:]
        y_val = y_train[-10000:]
        x_train = x_train[:-10000]
        y_train = y_train[:-10000]

        There's 2-3 line bits everywhere in this codebase that will be re-used in the context of the federated training. Re-write it if you need. Be patient with the overlaps.

        Add dataset, and dataset_labels as parameter (overload this method or make re-write)

        '''
        # x_train stores all of the train_images and y_train stores all the respective categories of each image, in the same order.
        # get cifar10-data first, and assign data and categorical labels as such
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

        x_train = x_train.reshape((-1, 32, 32, 3))
        x_test = x_test.reshape((-1, 32, 32, 3))

        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        history = self.model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))
        self.model.evaluate(x=x_test, y=y_test, verbose=0)

        self.model.save_weights('network.h5')
        print(history)

    def evaluate(self, image_set, label_set):
        '''Evaluate with test set, generally isolated to one client node for tensorflow-specific custom evaluation. Given we want to pass in custom image_set and custom image_labels.'''
        self.model.evaluate(x=image_set, y=label_set, verbose=0)
        return self.model

def main():
    # note that for each epoch_set we will iterate over each perturbation_epsilon and attack_type, defined in deploy.main
    graph = tf.compat.v1.get_default_graph()
    # do we use session = tf.Session() to instantiate a graph to execute our computation?
    # train network
    network = Network()
    # print(network.model.summary())
    network.build_compile_model()
    network.train()

    # evaluate model with cifar-10 data 
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    x_train = x_train.reshape((-1, 32, 32, 3))
    x_test = x_test.reshape((-1, 32, 32, 3))
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    network.evaluate(x_test, y_test)
