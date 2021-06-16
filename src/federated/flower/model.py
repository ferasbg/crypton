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

    def __init__(self, num_classes):

        self.num_classes = num_classes
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
        self.weight_decay_regularization = 0.003# batch norm, weight decay reg., fed optimizer, momentum for SGD --> how do they affect model given robust adversarial example
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
        model.add(Dense(self.num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros'))
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
        model.add(Dense(self.num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros'))
        # sgd(momentum=0.9), adam(lr=1e-2) -> helps weight regularization
        return model

    def train_model(self):
        # x_train stores all of the train_images and y_train stores all the respective categories of each image, in the same order.
        # get cifar10-data first, and assign data and categorical labels as such
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        history = self.model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))
        print(history)
        return self.model

    def evaluate_model(self, image_set, label_set):
        '''Evaluate with test set, generally isolated to one client node for tensorflow-specific custom evaluation. Given we want to pass in custom image_set and custom image_labels.'''
        self.model.evaluate(x=image_set, y=label_set, verbose=0)
        return self.model

# parser = argparse.ArgumentParser(description="Flower")
# parser.add_argument("--partition", type=int,
#                     choices=range(0, NUM_CLIENTS), required=True)
# args = parser.parse_args()
# create partition with train/test data per client; note that 600 images per client for 100 clients is convention; 300 images for 200 shards for 2 shards per client is another method and not general convention, but a test
# partition data (for a client) and pass to model

def load_partition_for_100_clients(idx: int):
    # 500/100 train/test split per partition e.g. per client
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    assert idx in range(100)

    return (
        # 5000/50000 --> 500/50000
        x_train[idx * 500: (idx + 1) * 500],
        y_train[idx * 500: (idx + 1) * 500],
    ), (
        x_test[idx * 100: (idx + 1) * 100],
        y_test[idx * 100: (idx + 1) * 100],
    )

def load_partition_for_10_clients(idx: int):
    # 500/100 train/test split per partition e.g. per client
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    assert idx in range(10)

    return (
        # 5000/50000 --> 500/50000
        x_train[idx * 5000: (idx + 1) * 5000],
        y_train[idx * 5000: (idx + 1) * 5000],
    ), (
        x_test[idx * 1000: (idx + 1) * 1000],
        y_test[idx * 1000: (idx + 1) * 1000],
    )

def main():
    graph = tf.compat.v1.get_default_graph()
    network = Network(num_classes=100)
    model = network.build_compile_model()
    print(network.model.summary())
    network.train_model()
    # evaluate with test data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    # partition data (for a client) and pass to model
    x_val, y_val = x_train[45000:50000], y_train[45000:50000]
    network.evaluate_model(x_val, y_val)
    
if __name__ == '__main__':
    main()