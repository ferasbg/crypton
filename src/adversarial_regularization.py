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
from typing import Dict, List, Tuple

# import art
# import cleverhans
import flwr as fl
import keras
# import matplotlib.pyplot as plt
import neural_structured_learning as nsl
import numpy
import numpy as np
import scipy
import sympy
import tensorflow as tf
# import tensorflow_datasets as tfds
# import tensorflow_federated as tff
# import tensorflow_privacy as tpp
# import tensorflow_probability as tpb
import tqdm
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import (FaultTolerantFedAvg, FedAdagrad, FedAvg,
                                  FedFSv1, Strategy, fedopt)
from keras import backend as K
from keras import optimizers, regularizers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.datasets import cifar10, cifar100
from keras.datasets.cifar10 import load_data
from keras.layers.core import Lambda
from keras.models import Input, Model, Sequential, load_model, save_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.backend import update
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.ops.gen_batch_ops import Batch
import warnings
warnings.filterwarnings("ignore")

class HParams(object):
    '''
    adv_multiplier: The weight of adversarial loss in the training objective, relative to the labeled loss.
    adv_step_size: The magnitude of adversarial perturbation.
    adv_grad_norm: The norm to measure the magnitude of adversarial perturbation.
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
        self.adv_grad_norm = adv_grad_norm  # "l2" or "infinity"

def build_compile_nsl_model(params: HParams, num_classes: int):
    input_layer = layers.Input(shape=(28, 28, 1), batch_size=None, name="image")
    conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
    batch_norm = layers.BatchNormalization()(conv1)
    dropout = layers.Dropout(0.3)(batch_norm)
    #  kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)
    conv2 = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform',padding='same')(dropout)
    maxpool1 = layers.MaxPool2D((2,2))(conv2)
    conv3 = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform',  padding='same')(maxpool1)
    conv4 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform',  padding='same')(conv3)
    maxpool2 = layers.MaxPool2D((2,2))(conv4)
    conv5 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform',  padding='same')(maxpool2)
    conv6 = layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_uniform',  padding='same')(conv5)
    maxpool3 = layers.MaxPool2D((2,2))(conv6)
    flatten = layers.Flatten()(maxpool3)
    dense1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flatten)
    # possibly remove defined kernel/bias initializer, but functional API will check for this and removes error before processing model architecture and config
    output_layer = layers.Dense(params.num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')(dense1)
    model = keras.Model(inputs=input_layer, outputs=output_layer, name='client_model')
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    adv_config = nsl.configs.make_adv_reg_config(multiplier=params.adv_multiplier, adv_step_size=params.adv_step_size, adv_grad_norm=params.adv_grad_norm)
    # AdvRegularization is a sub-class of tf.keras.Model, but it processes dicts instead for train and eval because of its decomposition approach for nsl
    adv_model = nsl.keras.AdversarialRegularization(model, label_keys=['label'], adv_config=adv_config, base_with_labels_in_features=True)
    adv_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return adv_model

# configure params object
parameters = HParams(num_classes=10, adv_multiplier=0.2,
                     adv_step_size=0.05, adv_grad_norm="infinity")

model = build_compile_nsl_model(params=parameters, num_classes=10)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test, y_test = x_train[-10000:], y_train[-10000:]
x_train = tf.cast(x_train, dtype=tf.float32)
x_test = tf.cast(x_test, dtype=tf.float32)
train_data = tf.data.Dataset.from_tensor_slices({'image': x_train, 'label': y_train}).batch(parameters.batch_size)
val_data = tf.data.Dataset.from_tensor_slices({'image': x_test, 'label': y_test}).batch(parameters.batch_size)
val_steps = x_test.shape[0] / parameters.batch_size

# type: History Callback object
history = model.fit(train_data, steps_per_epoch=1, epochs=5, batch_size=parameters.batch_size, verbose=1)
# these are the variables that are calculated and stored in the tf.keras.callbacks.History object
keys = history.history.keys() # dict_keys(['loss', 'sparse_categorical_crossentropy', 'sparse_categorical_accuracy', 'scaled_adversarial_loss'])
loss = history.history["loss"]
sparse_categorical_crossentropy = history.history["sparse_categorical_crossentropy"]
sparse_categorical_accuracy = history.history["sparse_categorical_accuracy"]
scaled_adversarial_loss = history.history["scaled_adversarial_loss"]

print("printing data stored in dict_keys from History object."  )
# it prints the loss for each epoch run
print(loss)
print(sparse_categorical_crossentropy)
print(sparse_categorical_accuracy)
print(scaled_adversarial_loss)


results = model.evaluate(val_data, verbose=1)

results = {
    "loss": results[0],
    "sparse_categorical_crossentropy": results[1],
    "sparse_categorical_accuracy": results[2],
    "scaled_adversarial_loss": results[3],
}

print("printing loss, accuracy: \n")
print(results["loss"])
print(results["sparse_categorical_accuracy"])

#base_client = Client(model, train_dataset=train_dataset_for_base_model, test_dataset=test_dataset_for_base_model, validation_steps=val_steps)
#flwr.client.start_keras_client(server_address="[::]:8080", client=base_client)
