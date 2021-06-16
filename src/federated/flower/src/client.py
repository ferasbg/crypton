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
from typing import Dict, List, Tuple

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
from flwr.common.typing import Weights
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import (  # FedProx; FedAdagrad helps convergence behavior which in turn helps optimize model robustness; fedOpt is configurable Adagrad for server-side optimizations for the server model e.g. trusted aggregator
    FaultTolerantFedAvg, FedAdagrad, FedAvg, FedFSv1, Strategy, fedopt)
from keras import backend as K
from keras import optimizers, regularizers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.datasets import cifar10
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
from PIL import Image
from tensorflow import keras
from typing import Dict, Tuple, cast
from tensorflow.python.keras.backend import update
from tensorflow.python.keras.engine.sequential import Sequential
from model import Network

NUM_CLIENTS = 100
CLIENT_LEARNING_RATE = 0.1
SERVER_LEARNING_RATE = 0.1
NUM_CLIENT_TRAIN_DATA = 500
NUM_CLIENT_TEST_DATA = 100
num_classes = 100

if __name__ == '__main__':
    class Client(flwr.client.NumPyClient):
        def __init__(self, model : Sequential, x_train, y_train, x_test, y_test):
            '''
            @model : pass model to flwr client.
            @gaussian_state : define if self.model should have a keras.layers.GaussianNoise layer during training. Pop the layer during evaluation, so check for this.
        
            '''
            
            super(Client, self).__init__()
            self.model = model 
            self.model_parameters = self.model.get_weights() 
            self.device_spec = tf.DeviceSpec(job="predict", device_type="GPU", device_index=0).to_string()
            self.gaussian_state = False
            if (self.gaussian_state == True):
                self.apply_gaussian_layer()

            # pass a loaded partition to be stored for the client
            self.x_train, self.y_train = x_train, y_train
            self.x_test, self.y_test = x_test, y_test
            
        def get_weights(self) -> Weights:
            return cast(Weights, self.model.get_weights())

        def fit(self, parameters, config):
            self.model.set_weights(parameters) 
            history = self.model.fit(self.x_train, self.y_train, epochs=5)
            num_examples_train = len(self.x_train)
            return self.model.get_weights(), num_examples_train 

        def evaluate(self, parameters, config):
            # the given image/label set is the defined partition rather than the entire dataset, since we iterate over client models with defined x_train, y_train, x_test, y_test sets
            self.model.set_weights(parameters)
            loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
            num_examples_test = len(self.x_test)
            return num_examples_test, loss, {"accuracy": accuracy}

        def apply_gaussian_layer(self):
            # the goal is to prevent adversarial over-fitting during this form of "augmentation"
            gaussian_layer = tf.keras.layers.GaussianNoise(stddev=0.2)
            client_model = Sequential()
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
            client_model.add(Dense(num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros'))
            optimizer = Adam(learning_rate=0.001) # lr=1e-2 
            # compile network with GaussianNoise layer
            client_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        optimizer=optimizer, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
            self.model = client_model 

    x_train, y_train = tf.keras.datasets.cifar100.load_data()
    x_test = x_train[-10000:]
    y_test = y_train[-10000:]

    model = Network(num_classes=num_classes).build_compile_model()
    network_client = Client(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    flwr.client.start_keras_client(server_address="[::]:8080", client=network_client)