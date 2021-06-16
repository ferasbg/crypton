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
    FaultTolerantFedAvg, FedAdagrad, FedAvg, FedFSv1, Strategy, fedopt) # can FedAdagrad be FaultTolerant e.g. accept_failures=True parameter
from keras import backend as K
from keras import optimizers, regularizers
from keras.applications.vgg16 import VGG16, preprocess_input
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
from model import Network, load_partition_for_10_clients, load_partition_for_100_clients

NUM_CLIENTS = 10 # 100
CLIENT_LEARNING_RATE = 0.1
SERVER_LEARNING_RATE = 0.1
NUM_CLIENT_TRAIN_DATA = 500
NUM_CLIENT_TEST_DATA = 100
num_classes = 100


class FederatedClient(flwr.client.NumPyClient):
    def __init__(self, model: Sequential, x_train, y_train, x_test, y_test):
        '''
        @model : pass model to flwr client.
        @gaussian_state : define if self.model should have a keras.layers.GaussianNoise layer during training. Pop the layer during evaluation, so check for this.
        '''
        super(FederatedClient, self).__init__()
        self.model = model
        # pass a loaded partition to be stored for the client
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_weights(self) -> Weights:
        # return cast(Weights, self.model.get_weights())
        raise Exception("Not implemented.")

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(self.x_train, self.y_train, batch_size=32, epochs=20)
        num_examples_train = len(self.x_train)
        weights = cast(Weights, self.model.get_weights())
        return weights, num_examples_train, history

    def evaluate(self, parameters, config):
        # the given image/label set is the defined partition rather than the entire dataset, since we iterate over client models with defined x_train, y_train, x_test, y_test sets
        self.model.set_weights(parameters)
        # define evaluation
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, batch_size=32, steps=5, verbose=1)
        num_examples_test = len(self.x_test)
        return num_examples_test, loss, {"accuracy": accuracy}

if __name__ == '__main__':
    model = Network(num_classes=num_classes).model
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    # split validation data
    x_test, y_test = x_train[45000:50000], y_train[45000:50000]
    client = FederatedClient(model, x_train=x_train,
                             y_train=y_train, x_test=x_test, y_test=y_test)
    flwr.client.start_numpy_client(server_address="[::]:8080", client=client)
