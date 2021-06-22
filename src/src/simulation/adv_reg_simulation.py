#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig

import argparse
import collections
import logging
import multiprocessing
import os
import pickle
import random
import sys
import threading
import time
import traceback
import warnings
from multiprocessing import Process
from typing import Dict, List, Tuple

import art
import cleverhans
import flwr as fl
import keras
import matplotlib.pyplot as plt
import neural_structured_learning as nsl
import numpy
import numpy as np
import scipy
import sympy
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_federated as tff
import tensorflow_privacy as tpp
import tensorflow_probability as tpb
import tqdm
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes
from flwr.server.client_proxy import ClientProxy
# FedAvg (Baseline); FedAdagrad (Comparable), FedOpt (Optimized FedAdagrad and Comparable)
from flwr.server.strategy import (FaultTolerantFedAvg, FedAdagrad, FedAvg,
                                  FedFSv1, Strategy, fedopt)
from keras import backend as K
from keras import optimizers, regularizers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.datasets import cifar10, cifar100
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
from tensorflow.keras import layers
from tensorflow.python.keras.backend import update
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.ops.gen_batch_ops import Batch

import dataset

# TODO: configure epochs, batch_size config based on HParams

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Tuple[np.ndarray, np.ndarray] --> [x_train, y_train]
# they store both train and test
DATASET = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]

class HParams(object):
    def __init__(self, num_classes, adv_multiplier, adv_step_size, adv_grad_norm):
        self.input_shape = [32, 32, 3]
        self.num_classes = num_classes
        self.conv_filters = [32, 64, 64, 128, 128, 256]
        self.kernel_size = (3, 3)
        self.pool_size = (2, 2)
        self.num_fc_units = [64]
        self.batch_size = 32
        self.epochs = 5
        self.adv_multiplier = adv_multiplier
        self.adv_step_size = adv_step_size
        self.adv_grad_norm = adv_grad_norm  # "l2" or "infinity"

def start_server(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start the server with a slightly adjusted FedAvg strategy."""
    strategy = FedAvg(min_available_clients=num_clients,
                      fraction_fit=fraction_fit)
    # Exposes the server by default on port 8080
    fl.server.start_server(strategy=strategy, config={
                           "num_rounds": num_rounds})

def start_client(dataset: DATASET) -> None:
    """Start a single client with the provided dataset."""
    # manually hardcode configs before extending to fixing import errors, parse_args for customization, etc just to see if there's function
    num_classes = 10
    # setup conditional to handle appending gaussian layer to base network; default=perturbations, next=gaussian noise + perturbations, ....
    gaussian_layer = keras.layers.GaussianNoise(stddev=0.2)
    parameters = HParams(num_classes=num_classes, adv_multiplier=0.2, adv_step_size=0.05, adv_grad_norm="infinity")
    input_layer = keras.Input(shape=(32,32,3), batch_size=None, name="image") # 32,32,1
    conv1 = layers.Conv2D(32, parameters.kernel_size, activation='relu', padding='same')(input_layer)
    batch_norm = layers.BatchNormalization()(conv1)
    dropout = layers.Dropout(0.3)(batch_norm)
    conv2 = layers.Conv2D(64, parameters.kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same')(dropout)
    maxpool1 = layers.MaxPool2D(parameters.pool_size)(conv2)
    conv3 = layers.Conv2D(64, parameters.kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same')(maxpool1)
    conv4 = layers.Conv2D(128, parameters.kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same')(conv3)
    maxpool2 = layers.MaxPool2D(parameters.pool_size)(conv4)
    conv5 = layers.Conv2D(128, parameters.kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same')(maxpool2)
    conv6 = layers.Conv2D(256, parameters.kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same')(conv5)
    maxpool3 = layers.MaxPool2D(parameters.pool_size)(conv6)
    flatten = layers.Flatten()(maxpool3)
    dense1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flatten)
    output_layer = layers.Dense(num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')(dense1)
    model = keras.Model(inputs=input_layer, outputs=output_layer, name='base_nsl_model')
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    adv_config = nsl.configs.make_adv_reg_config(multiplier=parameters.adv_multiplier, adv_step_size=parameters.adv_step_size, adv_grad_norm=parameters.adv_grad_norm)
    adv_model = nsl.keras.AdversarialRegularization(model, label_keys=['label'], adv_config=adv_config, base_with_labels_in_features=True)
    adv_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    (x_train, y_train), (x_test, y_test) = dataset

    # Define a Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):
            """Return current weights."""
            return model.get_weights()

        def fit(self, parameters, config):
            """Fit model and return new weights as well as number of training
            examples."""
            model.set_weights(parameters)
            # https://keras.io/api/models/model_training_apis/#fit-method
            # why is it that FitRes can process fit param of dict map but EvaluateRes cannot??
            # x_train is a set of np.ndarrays
            model.fit(x={"image": x_train, "label": y_train}, epochs=1,
                      batch_size=32, steps_per_epoch=3)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):
            """Evaluate using provided parameters."""
            model.set_weights(parameters)
            # x_test : <class 'tensorflow.python.framework.ops.EagerTensor'>
            # processing dict keys; or non-iterable "Tensor"
            # both of type np.ndarray; error with non-iterable EagerTensor given tuple of dict keys??
            # expected 2 e.g. the two variables to distribute the outputs to
            # dataset element processed and evaluate() function
            # this problem has to do with preprocessing of data and fitting it to the DATASET config in simulation
            # label_keys: A tuple of strings denoting which keys in the input features (a dict mapping keys to tensors) represent labels. This list should be 1-to-1 corresponding to the output of the base_model.
            loss, accuracy = model.evaluate(x_test, y_test)

            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8080", client=CifarClient())

def run_simulation(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start a FL simulation."""

    # This will hold all the processes which we are going to create
    processes = []

    # Start the server
    server_process = Process(
        target=start_server, args=(num_rounds, num_clients, fraction_fit)
    )
    server_process.start()
    processes.append(server_process)

    # Optionally block the script here for a second or two so the server has time to start
    time.sleep(2)

    # Load the dataset partitions
    partitions = dataset.load(num_partitions=num_clients)

    # Start all the clients
    for partition in partitions:
        # method callable target arg start_client
        client_process = Process(target=start_client, args=(partition,))
        client_process.start()
        processes.append(client_process)

    # Block until all processes are finished
    for p in processes:
        p.join()

if __name__ == "__main__":
    run_simulation(num_rounds=100, num_clients=10, fraction_fit=0.5)
