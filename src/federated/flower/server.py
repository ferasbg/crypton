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
from flwr.common.typing import Weights
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
IMAGE_TRANSFORMATION_STATE = False # assess configuration combinations of gaussian-distribution perturbation, image transformation/corruptions during training
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def main():
    # setup client
    num_classes = 100
    model = Network(num_classes=num_classes).build_compile_model()
    client = Client(model, defense_state=False)
    eval_fn = get_eval_fn(model)
    on_fit_config_fn = fit_config
    on_evaluate_config_fn = evaluate_config
    initial_parameters = model.get_weights()  # initial server parameters
    
    # hardcode strategy for now
    strategy = FedAvg(fraction_fit=0.3, fraction_eval=0.2, min_fit_clients=101, min_eval_clients=101, min_available_clients=110,
                        eval_fn=eval_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, initial_parameters=initial_parameters)
    
    # before specifying partitions and state of the network in training and test mode, let's first make this code functional
    flwr.server.start_server("[::]:8080", config={
                           "num_rounds": NUM_ROUNDS}, strategy=strategy)

if __name__ == '__main__':
    main()
