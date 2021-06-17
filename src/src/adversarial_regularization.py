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


class AdversarialRegularization(tf.keras.models.Model):
    '''

    References:
        - https://github.com/tensorflow/neural-structured-learning/blob/master/g3doc/tutorials/adversarial_keras_cnn_mnist.ipynb
    '''

    def __init__(self, num_classes, ):
        super(AdversarialRegularization, self).__init__()
        self.model = []
        self.gaussian_layer = keras.layers.GaussianNoise(stdev=0.2)
        self.num_classes = num_classes

    def create_adversarial_regularization_model():
        num_classes = 10
        model = Network(num_classes).build_compile_model()
        return model


class HParams(object):
    '''
    adv_multiplier: The weight of adversarial loss in the training objective, relative to the labeled loss.
    adv_step_size: The magnitude of adversarial perturbation.
    adv_grad_norm: The norm to measure the magnitude of adversarial perturbation.
    '''

    def __init__(self):
        self.input_shape = [32, 32, 3]
        self.num_classes = 10
        self.conv_filters = [32, 64, 64]
        self.kernel_size = (3, 3)
        self.pool_size = (2, 2)
        self.num_fc_units = [64]
        self.batch_size = 32
        self.epochs = 5
        self.adv_multiplier = 0.2
        self.adv_step_size = 0.2
        self.adv_grad_norm = 'infinity'


parameters = HParams()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_test, y_test = x_train[-10000:], y_train[-10000:]

IMAGE_INPUT_NAME = 'image'
IMAGE_LABEL_NAME = 'label'

def build_base_model(parameters):
    inputs = tf.keras.Input(shape=parameters.input_shape,
                            dtype=tf.float32, name=IMAGE_INPUT_NAME)
    x = inputs

    for i, num_filters in enumerate(parameters.conv_filters):
        x = tf.keras.layers.Conv2D(
            num_filters, parameters.kernel_size, activation='relu')(
                x)
        if i < len(parameters.conv_filters) - 1:
            # max pooling between convolutional layers
            x = tf.keras.layers.MaxPooling2D(parameters.pool_size)(x)

    x = tf.keras.layers.Flatten()(x)
    for num_units in parameters.num_fc_units:
        x = tf.keras.layers.Dense(num_units, activation='relu')(x)
    pred = tf.keras.layers.Dense(
        parameters.num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=pred)
    return model

model = build_base_model(parameters)
print(model.summary())
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=parameters.epochs)

results = model.evaluate(x_test, y_test, verbose=1)
print(results)
adv_config = nsl.configs.make_adv_reg_config(
    multiplier=parameters.adv_multiplier, adv_step_size=parameters.adv_step_size, adv_grad_norm=parameters.adv_grad_norm)

print(adv_config)
adversarial_regularized_model = nsl.keras.AdversarialRegularization(model, label_keys=[IMAGE_LABEL_NAME], adv_config=adv_config)
print(adversarial_regularized_model.summary())
adversarial_regularized_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
adversarial_regularized_model.fit(x_train, y_train, epochs=parameters.epochs)
results = adversarial_regularized_model.evaluate(x_test, y_test, verbose=1)
print(results)

## apply perturbations to CIFAR-100 dataset to evaluate regularized/un-regularized models
