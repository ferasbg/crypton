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

import art
import cleverhans
import flwr as fl
import keras
import matplotlib.pyplot as plt
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
from flwr.server.strategy import (  # FedProx; FedAdagrad helps convergence behavior which in turn helps optimize model robustness; fedOpt is configurable Adagrad for server-side optimizations for the server model e.g. trusted aggregator
    FaultTolerantFedAvg, FedAdagrad, FedAvg, FedFSv1, Strategy, fedopt)
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
from tensorflow.python.keras.backend import update
from tensorflow.python.keras.engine.sequential import Sequential

from model import Network


# tigher distribution
def get_larger_perturbation_epsilons():
    larger_perturbation_epsilons = [
        0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
    return larger_perturbation_epsilons

# "looser" distribution


def get_smaller_perturbation_epsilons():
    smaller_perturbation_epsilons = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return smaller_perturbation_epsilons


def compute_norm_bounded_perturbation(input_image, norm_type, perturbation_epsilon):
    '''Reuse for each additive perturbation type for all norm-bounded variants. Note that this is a norm-bounded ball e.g. additive and iterative perturbation attack. Specify the norm_type : l-2, l-inf, l-1'''
    if (norm_type == 'l-inf'):
        for pixel in input_image:
            pixel *= perturbation_epsilon + 0.5  # additive perturbation given vector norm
    elif (norm_type == 'l-2'):
        for pixel in input_image:
            pixel *= perturbation_epsilon + 0.5


def brightness_perturbation_norm(input_image):
    # applying image perturbations isn't in a layer, but rather before data is processed into the InputSpec and Input layer of the tff model
    sigma = 0.085
    brightness_threshold = 1 - sigma
    input_image = tf.math.scalar_mul(brightness_threshold, input_image)
    return input_image
