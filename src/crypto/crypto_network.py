import argparse
import collections
import logging
import os
import pickle
import random
import sys
import time
import warnings

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import tqdm
from keras import backend as K
from keras import optimizers, regularizers
from keras.applications.vgg16 import VGG16, preprocess_input
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

from crypto_utils import model_fn

warnings.filterwarnings('ignore')



class Network():
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
        self.weight_decay_regularization = 0.003 # us kernel_regularizer in Conv2d as iteration
        self.momentum = 0.05  # gradient descent convergence optimizer
        self.model = self.build_compile_model()

    def build_compile_model(self):
        # build layers of public neural network
        model = Sequential()
        # feature layers

        # input_shape maps exactly to Conv2d, maybe it should have more padding
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
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer, metrics=['accuracy'])
        return model


    def federated_train(self, batch_size, epochs, client_train_data, client_train_labels, client_validation_data, client_validation_labels):
        '''We want to use federated_train() for each client that runs local model.'''
        history = self.model.fit(client_train_data, client_train_labels, batch_size=batch_size, epochs=epochs, validation_data=(client_validation_data, client_validation_labels))
        self.model.evaluate(x=client_validation_data, y=client_validation_labels, verbose=0)
        self.model.save_weights('network.h5')
        print(history)
        return self.model


    def train(self):
        # get cifar10-data first, and assign data and categorical labels as such
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        x_train = x_train.reshape((-1, 32, 32, 3))
        x_test = x_test.reshape((-1, 32, 32, 3))

        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        history = self.model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))
        self.model.evaluate(x=x_test, y=y_test, verbose=0)

        self.model.save_weights('network.h5')
        print(history)
        

    @staticmethod
    def getClassificationState():
        return Network.classification_state

class CryptoNetwork(Network):
    """
        Deep Convolutional Neural Network With Federated Computation 

        Note that in real-world production scenario, we would have to analyze the states of our clients such that local training is possible since on-prem only works when their devices are on. Also note that data in the real-world is always messy and not clean as the datasets being used for this process.

        "There is a fixed set of K clients, each with a fixed local dataset. At the beginning of each round, a random fraction C of clients is selected, and the server sends the current global algorithm state to each of these clients (e.g., the current model parameters). We only select a fraction of clients for efficiency, as our experiments show diminishing returns for adding more clients beyond a certain point. Each selected client then performs local computation based on the global state and its local dataset, and sends an update to the server. The server then applies these updates to its global state, and the process repeats." (McMahan et. al)

        "The key consequence of this is that federated computations, by design, are expressed in a manner that is oblivious to the exact set of participants; all processing is expressed as aggregate operations on an abstract group of anonymous clients, and that group might vary from one round of training to another. The actual binding of the computation to the concrete participants, and thus to the concrete data they feed into the computation, is thus modeled outside of the computation itself."

        Raises:
        Returns:
        References:
        Examples:

    """

    def __init__(self):
        '''Build federated variant of convolutional network.'''
        super(CryptoNetwork, self).__init__()
        # get plaintext layers for network architecture, focus primarily on heavy dp and federated e.g. iterate on data processing to ImageDataGenerator and model.fit_generator() or model.fit()
        self.public_network = super().build_compile_model()
        self.input_spec = collections.OrderedDict(
            x=tf.TensorSpec(shape=[None, 32*32], dtype=tf.float32, name='pixels'),
            y=tf.TensorSpec(shape=[None, None], dtype=tf.int64, name='label')  
        )
        # tff wants new tff network created upon instantiation or invocation of method call
        self.crypto_network =  tff.learning.from_keras_model(self.public_network, input_spec=self.input_spec, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])

    def getTrainingData(self):
        (x_train, y_train), (x_test, y_test) = tff.simulation.datasets.cifar100.load_data()
        x_train = x_train.reshape((-1, 32, 32, 3))
        x_test = x_test.reshape((-1, 32, 32, 3))
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        return x_train, x_test 

    @staticmethod
    def client_optimizer_fn():
        return tf.keras.optimizers.SGD(learning_rate=0.02)

    @staticmethod
    def server_optimizer_fn():
        return tf.keras.optimizers.SGD(learning_rate=1.0)


if __name__ == '__main__':
    # setup crypto_network, federated_dataset, federated_clients, setup federated_eval() 
    crypto_network = CryptoNetwork()
    cifar_train, cifar_test = tff.simulation.datasets.cifar100.load_data()
    print(len(cifar_train.client_ids)) # 500 client ids for cifar-100
    print(cifar_train.element_type_structure) # (32,32,3)

    