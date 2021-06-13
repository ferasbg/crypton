#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig

import argparse
import collections
import logging
import os
import pickle
import random
import sys
import time
import warnings

import flwr
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_federated as tff
import tensorflow_privacy as tpp
import tensorflow_probability as tpb
import tqdm
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes
from keras import backend as K
from keras import optimizers, regularizers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.datasets.cifar10 import load_data
from keras.datasets.cifar100 import load_data
from keras.layers import (Activation, BatchNormalization, Conv2D,
                          Conv2DTranspose, Dense, Dropout, Flatten,
                          GaussianDropout, GaussianNoise, Input, MaxPool2D,
                          ReLU, Softmax, UpSampling2D)
from keras.layers.core import Lambda
from keras.models import Input, Model, Sequential, load_model, save_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from model import Network
from PIL import Image
from tensorflow import keras
from tensorflow.python.keras.engine.sequential import Sequential
from typing import Dict, List, Tuple

class Client(flwr.client.NumPyClient):
    def __init__(self, model : Sequential):
        super(Client, self).__init__()
        # can it be assumed that param order or count is negligible if they are optional
        self.model = model # model = Network().build_compile_model()
        self.model_parameters = self.network.get_weights() 
        self.device_spec = tf.DeviceSpec(job="predict", device_type="GPU", device_index=0).to_string()
        self.uncompiled_gaussian_network = []
        self.compiled_gaussian_network = []
        self.defense_state = False
        if (self.defense_state == True):
            # apply all defenses, which are configurable to assess what defenses directly relate to more robust adversarial examples and robust generalization and model convergence with respect to adaptive federated optimizer
            self.apply_defenses()
        
    def get_parameters(self, model: Sequential) -> List[np.ndarray]:
        return model.get_weights()

    # model.fit() 
    def fit(self, parameters, model: Sequential, x_train: list, y_train: list):
        # # set the weights of the Dense layer
        # model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=5)
        # model.get_weights() == self.get_parameters()
        return model.get_weights(), len(x_train)

    # model.evaluate()
    def evaluate(self, parameters, model, x_test, y_test):
        self.model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return len(x_test), loss, accuracy

    def set_defense_state(gaussian_state : bool) -> None:
        defense_state = gaussian_state
    
    def apply_defenses(self):
        # this function exists so we can write more defenses to modify both our model and the data passed to it
        # apply these transformations to modify the uncompiled plaintext model that will be passed into the crypto_network object inherited from CryptoNetwork. 
        self.apply_gaussian_layer()

    def apply_gaussian_layer(self):
        # the goal is to prevent adversarial over-fitting during this form of "augmentation"
        # note that if u rebuild the model and apply the gaussian noise layer, it MUST be done before it trains and makes inference in a federated/non-federated environment
        client_model = Sequential()
        gaussian_layer = tf.keras.layers.GaussianNoise(stddev=0.2)
        client_model.add(gaussian_layer)
        client_model.add(Conv2D(32, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        client_model.add(BatchNormalization())
        client_model.add(Dropout(0.3))
        client_model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        client_model.add(MaxPool2D((2, 2)))
        client_model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        client_model.add(Conv2D(128, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        client_model.add(MaxPool2D((2, 2)))
        client_model.add(Conv2D(128, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        client_model.add(Conv2D(256, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        client_model.add(MaxPool2D((2, 2)))
        client_model.add(Flatten())
        client_model.add(Dense(128, activation='relu',
                        kernel_initializer='he_uniform'))
        client_model.add(Dense(10, activation='softmax'))
        # have an uncompiled copy for custom (and adaptive -> to the heterogeneous data) federated optimizer
        self.uncompiled_gaussian_network = client_model
        
        optimizer = Adam(learning_rate=0.001) # lr=1e-2; general purpose 
        # compile network with GaussianNoise layer
        client_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer=optimizer, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        self.compiled_gaussian_network = client_model # Adam
