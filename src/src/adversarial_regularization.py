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
import neural_structured_learning as nsl
from PIL import Image
from tensorflow import keras
from tensorflow.python.keras.backend import update
from tensorflow.python.keras.engine.sequential import Sequential

from model import Network


class AdversarialRegularizationWrapper():
    '''
    Discussion:
        - formalize relationship between adversarial input generation for a client and its server-side evaluation-under-attack.
        - adversarial regularization is very useful for defining an explicit structure e.g. structural signals rather than single samples.
        - nsl-ar structured signals provides more fine-grained information not available in feature inputs.
        - We can assume training with robust adversarial examples makes it robust against adversarial perturbations during inference (eval), but how does this process fair when there's a privacy-specific step of using client models to locally train on this data and use a federated optimization technique for server-side evaluation? How can we utilize unsupervised/semi-supervised learning and these "structured signals" to learn hidden representations in perturbed or otherwise corrupted data (image transformations) with applied gaussian noise (these configurations exist to simulate a real-world scenario). We want to then formalize this phenomenon and the results between each experimental configuration.

    References:
        - https://github.com/tensorflow/neural-structured-learning/blob/master/g3doc/tutorials/adversarial_keras_cnn_mnist.ipynb
    '''

    def __init__(self, num_classes: int, model: Sequential):
        super(AdversarialRegularizationWrapper, self).__init__()
        self.model = model
        self.gaussian_layer = keras.layers.GaussianNoise(stdev=0.2)
        self.num_classes = num_classes

    def create_adversarial_regularization_model(num_classes: int):
        model = Network(num_classes).build_compile_model()
        return model


class HParams(object):
    '''
    adv_multiplier: The weight of adversarial loss in the training objective, relative to the labeled loss.
    adv_step_size: The magnitude of adversarial perturbation.
    adv_grad_norm: The norm to measure the magnitude of adversarial perturbation.

    Notes:
        - there are different regularization techniques, but keep technique constant
    '''

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
        self.adv_grad_norm = adv_grad_norm # "l2" or "infinity"


# configure params object
parameters = HParams(num_classes=100, adv_multiplier=0.2,
                     adv_step_size=0.2, adv_grad_norm="infinity")

# preprocess data into format they want
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test, y_test = x_train[-10000:], y_train[-10000:]

# IMAGE_INPUT_NAME = 'image'
# IMAGE_LABEL_NAME = 'label'


def build_uncompiled_model(parameters: HParams, num_classes : int):
    model = Sequential()
    model.add(Conv2D(32, parameters.kernel_size, activation='relu',
                     kernel_initializer='he_uniform', padding='same', input_shape=(32,32,3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(64, parameters.kernel_size, activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D(parameters.pool_size))
    model.add(Conv2D(64, parameters.kernel_size, activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, parameters.kernel_size, activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D(parameters.pool_size))
    model.add(Conv2D(128, parameters.kernel_size, activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(256, parameters.kernel_size, activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D(parameters.pool_size))
    model.add(Flatten())
    # model.add(Dense(128, activation='relu',
    #                 kernel_initializer='he_uniform'))
    model.add(Dense(num_classes, activation='softmax',
                    kernel_initializer='random_normal', bias_initializer='zeros'))
    return model

model = build_uncompiled_model(parameters, num_classes=10)
# model.fit(x_train, y_train, epochs=parameters.epochs)
# results = model.evaluate(x_test, y_test, verbose=1)

# possibly need to setup the same formatting scheme as specified in the tutorial

adv_config = nsl.configs.make_adv_reg_config(
    multiplier=parameters.adv_multiplier, adv_step_size=parameters.adv_step_size, adv_grad_norm=parameters.adv_grad_norm)

adv_model = nsl.keras.AdversarialRegularization(
    model, label_keys=('label', ), adv_config=adv_config)

adv_model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
adv_model.fit(x={'input': x_train, 'label': y_train}, batch_size=32, epochs=5, steps_per_epoch=5)
results = adv_model.evaluate(x_test, y_test, verbose=1)
print(results)
