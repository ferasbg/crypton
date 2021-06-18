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
from typing import Dict, List, NamedTuple, Tuple

import art
import cleverhans
import flwr
import flwr as fl
import keras
import matplotlib.pyplot as plt
import neural_structured_learning as nsl
import numpy
import numpy as np
import scipy
import sympy
import tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_federated as tff
import tensorflow_privacy as tpp
import tensorflow_probability as tpb
import tqdm
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes
from flwr.common.typing import Weights
from flwr.server.client_proxy import ClientProxy
# FedAvg (Baseline); FedAdagrad (Comparable), FedOpt (Optimized FedAdagrad and Comparable)
from flwr.server.strategy import (  # FedProx; FedAdagrad helps convergence behavior which in turn helps optimize model robustness; fedOpt is configurable Adagrad for server-side optimizations for the server model e.g. trusted aggregator
    FaultTolerantFedAvg, FedAdagrad, FedAvg, FedFSv1, Strategy, fedopt)
from keras import backend as K
from keras import optimizers, regularizers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.datasets import cifar10, cifar100
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
from tensorflow.keras import layers
from tensorflow.python.keras.backend import update
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.ops.gen_batch_ops import Batch

from adversarial_regularization import (AdversarialRegularizationWrapper,
                                        HParams, build_adv_reg_model,
                                        build_uncompiled_nsl_model)
from client import FederatedClient
from model import Network
from server import evaluate_config, fit_config, get_eval_fn

def make_training_data_iid(x_train):
    # IID: data is shuffled, then partitioned into 100 clients with 500 train and 100 test examples per client
    return x_train

def make_training_data_non_iid(x_train):
    # Non-IID: first sort the data, divide it into 200 shards of size 300 and assign 100 clients 2 shards
    return x_train

def create_clients(num_classes : int, num_clients : int, client_networks : list, clients : list):
    '''
    @param num_classes : int depends on the CIFAR dataset used.

    '''
    # creates 100 client models and adds the sequential models into a list, the flwr.client objects in their own list
    for i in range(num_clients):
        # setup base (unconfigured, compiled) client models (default configs)
        client_network = Network(num_classes=num_classes).build_compile_model() 
        client_networks.append(client_network)
        client = FederatedClient(client_network, defense_state=False)
        clients.append(client)
        assert len(client_networks) == 100

def adaptive_federated_adagrad(client_model : Sequential):
    federated_adagrad_strategy = FedAdagrad(
        fraction_fit=0.3,
        fraction_eval=0.2,
        min_fit_clients=101,
        min_eval_clients=101,
        min_available_clients=110,
        eval_fn=get_eval_fn(client_model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        accept_failures=False,
        initial_parameters=client_model.get_weights(),
        tau=1e-9,
        eta=1e-1,
        eta_l=1e-1
    )

    return federated_adagrad_strategy

def build_compile_client_model(adversarial_regularization_state : bool, num_classes : int):
    if (adversarial_regularization_state == True):
        parameters = HParams(num_classes=10, adv_multiplier=0.2, adv_step_size=0.05, adv_grad_norm="infinity")
        model = build_uncompiled_nsl_model(parameters=parameters, num_classes=num_classes)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        adv_config = nsl.configs.make_adv_reg_config(multiplier=parameters.adv_multiplier, adv_step_size=parameters.adv_step_size, adv_grad_norm=parameters.adv_grad_norm)
        adv_model = nsl.keras.AdversarialRegularization(model, adv_config=adv_config)
        
        adv_model.compile(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return adv_model
    
    elif (adversarial_regularization_state == False):
        # change norm type and adv_step_size iteratively as its own config per exp
        parameters = HParams(num_classes=10, adv_multiplier=0.2,
                            adv_step_size=0.05, adv_grad_norm="infinity")
        model = build_uncompiled_nsl_model(parameters, num_classes=num_classes)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        return model