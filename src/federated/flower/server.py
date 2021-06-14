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
from adversarial import *
from adversarial import Perturbation
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import (  # FedProx; FedAdagrad helps convergence behavior which in turn helps optimize model robustness; fedOpt is configurable Adagrad for server-side optimizations for the server model e.g. trusted aggregator
    FaultTolerantFedAvg, FedAdagrad, FedAvg, FedFSv1, Strategy, fedopt)
from formal_robustness import *
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
from metrics import FederatedMetrics
from model import Network
from PIL import Image
from tensorflow import keras
from tensorflow.python.keras.backend import update
from tensorflow.python.keras.engine.sequential import Sequential

from client import Client, FederatedClient
from flower.client import *
from flower.utils import *

# ENVIRONMENT VARIABLES
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
NUM_CLIENT_TRAIN_DATA = 500
NUM_CLIENT_TEST_DATA = 100
CLIENT_GAUSSIAN_STATE = False
FEDERATED_OPTIMIZATION_STATE = False
FEDERATED_STRATEGY = [] # federated strategy setting to change  
PERTURBATION_STATE = False
TRAIN_STATE = False
TEST_STATE = False
CIFAR_10_STATE = False  
CIFAR_100_STATE = True
# assess configuration combinations of gaussian-distribution perturbation, image transformation/corruptions during training
IMAGE_TRANSFORMATION_STATE = False
CONFIGURATION_COUNTER = 0
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

clients = []
# compiled networks use Adam optimizer, but it's still different than the strategy
client_networks = []
# we can iteratively use dataset partitions in train/test per client 
client_train_image_dataset = []
client_train_label_dataset = []
client_test_image_dataset = []
client_test_label_dataset = []
uncompiled_client_networks = []
num_classes = 0

# 100 classes/label options
if (CIFAR_100_STATE == True):
    x_train, y_train = tf.keras.datasets.cifar100.load_data()
    x_test, y_test = x_train[50000:60000], y_train[50000:60000]
    num_classes = 100

# 10 classes/label options
if (CIFAR_10_STATE == True):
    # partitions between clients will also be different since the total set consists of
    x_train, y_train = tf.keras.datasets.cifar10.load_data()
    x_test, y_test = x_train[50000:60000], y_train[50000:60000]
    # make sure that the Dense layer reads in 10 classes for CIFAR-10
    num_classes = 10

def create_clients():
    for i in range(NUM_CLIENTS):
        # setup base (unconfigured, compiled) client models (default configs)
        client_network = Network(num_classes=num_classes).build_compile_model() 
        client = Client(client_network, defense_state=False)
        # thinking abt object variable access
        client_networks.append(client_network)
        assert len(client_networks) == 100

def partition_client_data():
    # 500 images per client, 100 clients --> that's 50,000 images for TRAIN, then 100 images per client for 100 clients for TEST
    # note that partitioning works the same way for both Cifar-10 and Cifar-100
    for client_network in client_networks:
        # execute partitions for train/test sets
        client_train_image_dataset[client_network] = x_train[:NUM_CLIENT_TRAIN_DATA]
        client_train_label_dataset[client_network] = y_train[:NUM_CLIENT_TRAIN_DATA]
        client_test_image_dataset[client_network] = x_test[:NUM_CLIENT_TEST_DATA]
        client_test_label_dataset[client_network] = x_test[:NUM_CLIENT_TEST_DATA]
    
    for x in range(len(client_train_image_dataset)):
        assert len(client_train_image_dataset[x]) == 500
    
    for y in range(len(client_train_label_dataset)):
        assert len(client_train_label_dataset[x]) == 500

def configure_client_models():
    # make the client models gaussian during training, then set the mode off by removing the layer from the stack without resetting the model parameters
    if (CLIENT_GAUSSIAN_STATE):
        for client_network in client_networks:
            model = client_networks[client_network]    
            model.update_defense_state(CLIENT_GAUSSIAN_STATE)

def configure_dataset():
    # preprocess data with transformations for train set
    if (IMAGE_TRANSFORMATION_STATE):
        # apply specific perturbations and update the norm values based on the ROUND, so every ROUND, update the norm value on a particular norm_type
        for i in range(len(x_train)):
            image = x_train[i]
            image = Perturbation.apply_image_degradation(image)
            x_train[i] = image

def configure_testing_process():
    # configure the perturbation types to use, norm type, and epsilon value per ROUND iteration
    pass

def get_eval_fn(model, test_image_dataset, test_label_dataset):
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

    strategy = FedAvg(fraction_fit=fraction_fit, fraction_eval=fraction_eval, min_fit_clients=min_fit_clients, min_eval_clients=min_eval_clients, min_available_clients=min_available_clients,
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
    # batches in validation/eval
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}

def get_federated_strategy(model : Sequential, option : int):
    # get_federated_strategy(int option)
    if (FEDERATED_OPTIMIZATION_STATE == 0):
        FEDERATED_STRATEGY = federated_averaging(model)

    if (FEDERATED_OPTIMIZATION_STATE == 1):
        FEDERATED_STRATEGY = adaptive_federated_optimization_adagrad(model)

    return FEDERATED_STRATEGY

def main():
    '''
        Todo:
            - get base case working e.g. the vanilla FedAvg for 100 clients; client grads get flushed after the training/test iteration is over anyway
            - measure for how adaptive federated optimizers help improve model convergence and consequently their robustness to adversarial examples in a federated setting
            - test different federated strategies and optimizers and configure them as such because fitting the optimizer to the type of data given that we have solved adversarial overfitting contributes to even greater robustness; here we can create all the relationships between what really formulates the model's expected behavior and we can measure for these changes by popping out particular components in the layer stack of the client network itself; nominal metrics against adversarial examples are a leading indicator of how they map to formal robustness
            - test given configurable strategies so FedAvg, FedAdagrad, FTFed; client_side_lr is not 0.02 which is interesting; need 100 clients to "simulate" how much data each user is expected to generally have; a more realistic situation would be 1000-10000 clients with 10-50 to 1-5 images each respectively per client size
    '''

    # setup experiment
    create_clients()
    partition_client_data()
    # core computation
    
    # loop involves partitioned dataset tuple per client as well as respective strategy that iterates over the clients
    federated_strategy = get_federated_strategy(client_networks[0], 0)    
    fl.server.start_server("[::]:8080", config={
                           "num_rounds": NUM_ROUNDS}, strategy=federated_strategy)

if __name__ == '__main__':
    main()
