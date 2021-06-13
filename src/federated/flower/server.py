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

import flwr as fl
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
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import (  # FedProx; FedAdagrad helps convergence behavior which in turn helps optimize model robustness; fedOpt is configurable Adagrad for server-side optimizations for the server model e.g. trusted aggregator
    FaultTolerantFedAvg, FedAdagrad, FedAvg, FedFSv1, Strategy, fedopt)
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
from model import Network
from PIL import Image
from tensorflow import keras
from tensorflow.python.keras.backend import update
from tensorflow.python.keras.engine.sequential import Sequential

from client import Client, FederatedClient

NUM_CLIENTS = 100
BATCH_SIZE = 32
NUM_EPOCHS = 10
CLIENT_LEARNING_RATE = 0.02
SERVER_LEARNING_RATE = 1.0
NUM_ROUNDS = 10
CLIENTS_PER_ROUND = 10
NUM_EXAMPLES_PER_CLIENT = 500
CIFAR_SHAPE = (32, 32, 3)
TOTAL_FEATURE_SIZE = 32 * 32 * 3
CLIENT_EPOCHS_PER_ROUND = 1
NUM_CLIENT_DATA = 500
CLIENT_GAUSSIAN_STATE = False
FEDERATED_OPTIMIZATION_STATE = False
PERTURBATION_STATE = False
TRAIN_STATE = False
TEST_STATE = False
CIFAR_10_STATE = False  # not right now
CIFAR_100_STATE = True
# merge image corruption and image transformations into 1 transformation state, since transformation implies image perturbation
IMAGE_TRANSFORMATION_STATE = False
CONFIGURATION_COUNTER = 0
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# each client_network derives from its respective client object, so client_network == client.model
clients = []
# are we going to just have uncompiled models because they'd have to specify the strategy to use and compiled models already use Adam
client_networks = []
# the orchestration of the dataset itself should depend on what dataset we are using
client_train_image_dataset = []
client_train_label_dataset = []
client_test_image_dataset = []
client_test_label_dataset = []
uncompiled_client_networks = []

# 100 classes/label options
if (CIFAR_100_STATE == True):
    x_train, y_train = tf.keras.datasets.cifar100.load_data()
    x_test, y_test = x_train[50000:60000], y_train[50000:60000]

# 10 classes/label options
if (CIFAR_10_STATE == True):
    # partitions between clients will also be different since the total set consists of
    x_train, y_train = tf.keras.datasets.cifar10.load_data()
    x_test, y_test = x_train[50000:60000], y_train[50000:60000]


def create_clients():
    for i in range(NUM_CLIENTS):
        # setup base (unconfigured) client models (defaults)
        client = Client(defense_state=False)
        uncompiled_client_network = client.build_uncompiled_model()
        client_network = client.build_compile_model()
        uncompiled_client_networks.append(uncompiled_client_network)
        clients.append(client)
        client_networks.append(client_network)


def partition_client_data():
    # 500 images per client, 100 clients --> that's 50,000 images for TRAIN, then 100 images per client for 100 clients for TEST
    # note that partitioning works the same way for both Cifar-10 and Cifar-100
    for client_network in client_networks:
        # execute partitions for train/test sets
        client_train_image_dataset[client_network] = x_train[:500]
        client_train_label_dataset[client_network] = y_train[:500]
        client_test_image_dataset[client_network] = x_test[:100]
        client_test_label_dataset[client_network] = x_test[:100]


def configure_training_process():
    for client_network in client_networks:
        # client_network : Client
        # assumption: CONFIGURATION_COUNTER applies to all the client models during each round rather than a subset
        if (CONFIGURATION_COUNTER == 0):
            # do configuration 1; set the client configs based on the config u want for all client models per (ROUND) iteration
            CLIENT_GAUSSIAN_STATE = False
            client_network.update_defense_state(CLIENT_GAUSSIAN_STATE)
            IMAGE_TRANSFORMATION_STATE = False
            # update PARTITION SPECIFIC TO THE CLIENT; DO NOT FORGET TO FLUSH THE STATE OF THE PARTITION DATA BY SETTING THE DATASET BACK TO ORIGINAL STATE
            FEDERATED_OPTIMIZATION_STATE = False
            CONFIGURATION_COUNTER += 1

        if (CONFIGURATION_COUNTER == 2):
            CLIENT_GAUSSIAN_STATE = False
            client_network.update_defense_state(CLIENT_GAUSSIAN_STATE)
            IMAGE_TRANSFORMATION_STATE = True
            FEDERATED_OPTIMIZATION_STATE = False
            CONFIGURATION_COUNTER += 1

        if (CONFIGURATION_COUNTER == 6):
            CLIENT_GAUSSIAN_STATE = True
            client_network.update_defense_state(CLIENT_GAUSSIAN_STATE)
            IMAGE_TRANSFORMATION_STATE = True
            FEDERATED_OPTIMIZATION_STATE = False
            CONFIGURATION_COUNTER += 1

        if (CONFIGURATION_COUNTER == 4):
            CLIENT_GAUSSIAN_STATE = True
            client_network.update_defense_state(CLIENT_GAUSSIAN_STATE)
            IMAGE_TRANSFORMATION_STATE = False
            FEDERATED_OPTIMIZATION_STATE = False
            CONFIGURATION_COUNTER += 1

        if (CONFIGURATION_COUNTER == 5):
            CLIENT_GAUSSIAN_STATE = True
            client_network.update_defense_state(CLIENT_GAUSSIAN_STATE)
            IMAGE_TRANSFORMATION_STATE = False
            FEDERATED_OPTIMIZATION_STATE = True
            CONFIGURATION_COUNTER += 1

        if (CONFIGURATION_COUNTER == 3):
            CLIENT_GAUSSIAN_STATE = False
            client_network.update_defense_state(CLIENT_GAUSSIAN_STATE)
            IMAGE_TRANSFORMATION_STATE = True
            FEDERATED_OPTIMIZATION_STATE = True
            CONFIGURATION_COUNTER += 1

        if (CONFIGURATION_COUNTER == 7):
            CLIENT_GAUSSIAN_STATE = True
            client_network.update_defense_state(CLIENT_GAUSSIAN_STATE)
            IMAGE_TRANSFORMATION_STATE = True
            FEDERATED_OPTIMIZATION_STATE = True
            CONFIGURATION_COUNTER += 1

        if (CONFIGURATION_COUNTER == 1):
            # configuration 2
            CLIENT_GAUSSIAN_STATE = False
            client_network.update_defense_state(CLIENT_GAUSSIAN_STATE)
            IMAGE_TRANSFORMATION_STATE = False
            FEDERATED_OPTIMIZATION_STATE = True
            CONFIGURATION_COUNTER += 1


def configure_testing_process():
    pass


def get_eval_fn(model, test_image_dataset, test_label_dataset):
    # nested callable function for internal library backend
    # return evaluation function for server-side evaluation
    def evaluate(weights: fl.common.Weights):
        model.set_weights(weights)
        loss, accuracy = model.evaluate(test_image_dataset, test_label_dataset)
        return loss, {"accuracy:": accuracy}
    return evaluate



def federated_averaging(model: Sequential, fraction_fit=0.3, fraction_eval=0.2, min_fit_clients=101, min_eval_clients=101, min_available_clients=110):
    eval_fn = get_eval_fn(model)
    on_fit_config_fn = fit_config
    on_evaluate_config_fn = evaluate_config
    initial_parameters = model.get_weights()  # initial server parameters

    strategy = FedAvg(fraction_fit=0.3, fraction_eval=fraction_eval, min_fit_clients=min_fit_clients, min_eval_clients=min_eval_clients, min_available_clients=min_available_clients,
                      eval_fn=eval_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, initial_parameters=initial_parameters)
    return strategy


def adaptive_federated_optimization_adagrad(model: Sequential, fraction_fit=0.3, fraction_eval=0.2, min_fit_clients=101, min_eval_clients=101, min_available_clients=110, adaptability_rate=1e-9, accept_failures=False, server_side_lr=1e-1, client_side_lr=1e-1):
    strategy = FedAdagrad(
        fraction_fit=fraction_fit,
        fraction_eval=fraction_eval,
        min_fit_clients=min_fit_clients,
        min_eval_clients=min_eval_clients,
        min_available_clients=min_available_clients,
        # require function
        eval_fn=get_eval_fn(model),
        # callable methods
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        accept_failures=accept_failures,
        initial_parameters=model.get_weights(),
        tau=adaptability_rate,
        eta=server_side_lr,
        eta_l=client_side_lr
    )
    return strategy


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


def main():
    '''

    Todo:
        - measure for how adaptive federated optimizers help improve model convergence and consequently their robustness to adversarial examples in a federated setting
        - test different federated strategies and optimizers and configure them as such because fitting the optimizer to the type of data given that we have solved adversarial overfitting contributes to even greater robustness; here we can create all the relationships between what really formulates the model's expected behavior and we can measure for these changes by popping out particular components in the layer stack of the client network itself; nominal metrics against adversarial examples are a leading indicator of how they map to formal robustness
        - test given configurable strategies so FedAvg, FedAdagrad, FTFed; client_side_lr is not 0.02 which is interesting; need 100 clients to "simulate" how much data each user is expected to generally have; a more realistic situation would be 1000-10000 clients with 10-50 to 1-5 images each respectively per client size
    '''

    # setup experiment
    create_clients()
    partition_client_data()
    configure_training_process()
    configure_testing_process()
    # core computation

    # uncompiled keras layers meet configurable federated optimizer and aggregation method(s)
    federated_averaging_strategy = federated_averaging()
    federated_adagrad = adaptive_federated_optimization_adagrad()
    fl.server.start_server("0.0.0.0:8080", config={
                           "num_rounds": NUM_ROUNDS}, strategy=federated_averaging_strategy)

    # get metrics (nominal --> formal; write setters)


if __name__ == '__main__':
    main()
