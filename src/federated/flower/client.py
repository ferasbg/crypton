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
from tensorflow.python.keras.backend import update
from tensorflow.python.keras.engine.sequential import Sequential
from model import Network

NUM_CLIENTS = 100
BATCH_SIZE = 32
NUM_EPOCHS = 10
CLIENT_LEARNING_RATE = 0.1
SERVER_LEARNING_RATE = 0.1
NUM_ROUNDS = 10
CLIENTS_PER_ROUND = 10
NUM_EXAMPLES_PER_CLIENT = 500
CIFAR_SHAPE = (32, 32, 3)
TOTAL_FEATURE_SIZE = 32 * 32 * 3
CLIENT_EPOCHS_PER_ROUND = 1
NUM_CLIENT_TRAIN_DATA = 500
NUM_CLIENT_TEST_DATA = 100
CLIENT_GAUSSIAN_STATE = False
FEDERATED_OPTIMIZATION_STATE = False
PERTURBATION_STATE = False
TRAIN_STATE = False
TEST_STATE = False
CIFAR_10_STATE = False  
CIFAR_100_STATE = True
IMAGE_TRANSFORMATION_STATE = False
clients = []
client_networks = []
client_train_image_dataset = []
client_train_label_dataset = []
client_test_image_dataset = []
client_test_label_dataset = []
uncompiled_client_networks = []
num_classes = 100

class Client(flwr.client.NumPyClient):
    def __init__(self, model : Sequential, gaussian_state : bool, x_train, y_train, x_test, y_test):
        '''
        @model : pass model to flwr client.
        @gaussian_state : define if the network should have a keras.layers.GaussianNoise layer during training. Pop the layer during evaluation, so check for this.
     
        '''
        
        super(Client, self).__init__()
        self.model = model 
        self.model_parameters = self.model.get_weights() 
        self.device_spec = tf.DeviceSpec(job="predict", device_type="GPU", device_index=0).to_string()
        self.compiled_gaussian_network = []
        self.gaussian_state = gaussian_state
        if (self.gaussian_state == True):
            self.apply_gaussian_layer()

        # pass a loaded partition to be stored for the client
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        
    def get_parameters(self) -> List[np.ndarray]:
        return self.model.get_weights()

    def fit(self, parameters, config):
        # x_train and y_train are are the partitioned train image/label sets for the CLIENT; it's not the entire CIFAR-10(0) dataset 
        parameters = self.model.set_weights(parameters) 
        batch_size : int = config["batch_size"]
        epochs : int = config["local epochs"]       
        # hyper-parameters of round; can we keep batch_size and epochs per round constant? Check fit_config to see what specs they have given the round number
        history = self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, steps_per_epoch=5, validation_split=0.1)
        # return updated model parameters (weights) and results
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }

        num_examples_train = len(self.x_train)

        return parameters, num_examples_train, results

    def evaluate(self, parameters, config):
        # the given image/label set is the defined partition rather than the entire dataset, since we iterate over client models with defined x_train, y_train, x_test, y_test sets
        self.model.set_weights(parameters)
        steps : int = config["val_steps"]
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}

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
        optimizer = Adam(learning_rate=0.001) # lr=1e-2 
        # compile network with GaussianNoise layer
        client_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer=optimizer, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        self.compiled_gaussian_network = client_model # Adam
    

def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (
        x_train[idx * 5000 : (idx + 1) * 5000],
        y_train[idx * 5000 : (idx + 1) * 5000],
    ), (
        x_test[idx * 1000 : (idx + 1) * 1000],
        y_test[idx * 1000 : (idx + 1) * 1000],
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    args = parser.parse_args()

    # create a client
    model = Network(num_classes=num_classes)
    model = model.build_compile_model()
    # dataset
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    client = Client(model, gaussian_state=False, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    flwr.client.start_numpy_client("[::]:8080", client=client)

if __name__ == '__main__':
    main()
