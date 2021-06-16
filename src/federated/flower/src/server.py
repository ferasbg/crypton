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

from model import Network

os.environ["TF_CPP_MIN_LG_LEVEL"] = "3"

def get_eval_fn(model):
    (x_train, y_train), _ = tf.keras.datasets.cifar100.load_data()
    x_val, y_val = x_train[45000:50000], y_train[45000:50000]

    # return evaluation function for server-side evaluation
    def evaluate(weights: Weights):
        model.set_weights(weights)
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, {"accuracy:": accuracy}
    return evaluate

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

if __name__ == '__main__':
    flwr.server.start_server("[::]:8080", config={
                           "num_rounds": 3})

