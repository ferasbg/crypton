import argparse
import collections
import logging
import os
import pickle
import random
import sys
import time
import warnings

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import tqdm
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
from model import *
from model import Network
from PIL import Image
from tensorflow import keras


NUM_CLIENTS = 100
BATCH_SIZE = 20
NUM_EPOCHS = 10
CLIENT_LEARNING_RATE = 0.02
SERVER_LEARNING_RATE = 1.0
NUM_ROUNDS = 20
CLIENTS_PER_ROUND = 2
NUM_EXAMPLES_PER_CLIENT = 500
CIFAR_SHAPE = (32,32,3)
TOTAL_FEATURE_SIZE = 32 * 32 * 3
CLIENT_EPOCHS_PER_ROUND = 1
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
client_dataset = []

def model_fn():
    # build layers of public neural network to pass into tff constructor as plaintext model
    model = Sequential()
    # feature layers
    model.add(tf.keras.Input(shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    # classification layers
    model.add(Dense(128, activation='relu',
                    kernel_initializer='he_uniform'))
    # 10 output classes possible
    model.add(Dense(10, activation='softmax'))

    input_spec = collections.OrderedDict(
        x=tf.TensorSpec([None, 32, 32, 3], tf.float32),
        y=tf.TensorSpec([None, 1], tf.int32)
    )
    # tff wants new tff network created upon instantiation or invocation of method call, uncompiled
    crypto_network = tff.learning.from_keras_model(model, input_spec=input_spec, loss=tf.keras.losses.SparseCategoricalCrossentropy(
    ), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return crypto_network


def client_optimizer_fn():
    # client optimizer_fn updates local client model while server_optimizer_fn applies the averaged update to the global model in the server
    # implementation: client_optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.02)
    return tf.keras.optimizers.SGD(learning_rate=0.02)


def server_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=1.0)


def make_federated_eval():
    # takes a model function and returns a single federated computation for federated evaluation of models, since evaluation is not stateful.
    federated_eval = tff.learning.build_federated_evaluation(model_fn)
    return federated_eval


cifar_train, cifar_test = tff.simulation.datasets.cifar100.load_data()
sample_clients = cifar_train.client_ids[0:NUM_CLIENTS]


def preprocess_dataset(dataset):
    def map_fn(element):
        return collections.OrderedDict(
            x=tf.cast(tf.reshape(element['image'],
                                 shape=(-1, 32, 32, 3)), tf.float32),
            y=tf.cast(tf.reshape(element['label'], shape=(-1, 1)), tf.int32),
        )

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).map(map_fn).prefetch(PREFETCH_BUFFER)


def setup_federated_data(client_data, client_ids, federated_train_data=[]):
    for x in client_ids:
        client_node_data = preprocess_dataset(
            client_data.create_tf_dataset_for_client(x))
        federated_train_data.append(client_node_data)
    return federated_train_data


def make_federated_data(client_data, client_ids):
    return [preprocess_dataset(client_data.create_tf_dataset_for_client(x) for x in client_ids)]


def federated_train_generator():
    return [(federated_train_data[x] for x in federated_train_data)]


def evaluate(server_state, federated_dataset):
    network = Network()
    network = network.build_uncompiled_model()
    network.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[
                    tf.keras.metrics.SparseCategoricalAccuracy()])
    network.set_weights(server_state)  # communication
    network.evaluate(federated_dataset)  # test model (client/server)
    

@tff.tf_computation
def server_init():
    model = model_fn()
    return model.trainable_variables


@tff.federated_computation
def initialize_fn():
    return tff.federated_value(server_init(), tff.SERVER)


@tf.function
def client_update(model, dataset, server_weights, client_optimizer):
    """Performs training (using the server model weights) on the client's dataset."""
    # Initialize the client model with the current server weights.
    client_weights = model.trainable_variables
    # Assign the server weights to the client model.
    tf.nest.map_structure(lambda x, y: x.assign(y),
                          client_weights, server_weights)

    # Use the client_optimizer to update the local model.
    for batch in dataset:
        with tf.GradientTape() as tape:
            # Compute a forward pass on the batch of data
            outputs = model.forward_pass(batch)
            # Compute the corresponding gradient
            grads = tape.gradient(outputs.loss, client_weights)
            grads_and_vars = zip(grads, client_weights)
            # Apply the gradient using a client optimizer.
            client_optimizer.apply_gradients(grads_and_vars)

    return client_weights


@tf.function
def server_update(model, mean_client_weights):
    """Updates the server model weights as the average of the client model weights."""
    model_weights = model.trainable_variables
    # Assign the mean client weights to the server model.
    tf.nest.map_structure(lambda x, y: x.assign(y),
                          model_weights, mean_client_weights)
    return model_weights


# define types to properly decorate tff functions
dummy_model = model_fn()
model_weights_type = server_init.type_signature.result
tf_dataset_type = tff.SequenceType(dummy_model.input_spec)
federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)


@tff.tf_computation(tf_dataset_type, model_weights_type)
def client_update_fn(tf_dataset, server_weights):
    model = model_fn()
    client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    return client_update(model, tf_dataset, server_weights, client_optimizer)


@tff.tf_computation(model_weights_type)
def server_update_fn(mean_client_weights):
    model = model_fn()
    return server_update(model, mean_client_weights)

def create_crypto_network():
    public_network = Network().build_compile_model()
    tff_network = tff.learning.from_keras_model(public_network, input_spec=get_input_spec(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return tff_network

@tff.federated_computation(federated_server_type, federated_dataset_type)
def next_fn(server_weights, federated_dataset):
    '''Compute client-to-server update and server-to-client update between nodes.
    Note, the server state stores the global model's gradients and its weights respectively given tff model state dict and optimizer_state e.g. vars of optimizer
    '''
    # broadcast server weights to client nodes for local model set
    server_weights_at_client = tff.federated_broadcast(server_weights)
    client_weights = tff.federated_map(
        client_update_fn, (federated_dataset, server_weights_at_client))

    # server averages client updates
    mean_client_weights = tff.federated_mean(client_weights)

    # server updates its model with average of the clients' updated gradients
    server_weights = tff.federated_map(server_update_fn, mean_client_weights)
    return server_weights


def client_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=0.02)


def get_tff_training_data():
    # returns data and corresponding labels
    (x_train, y_train), (x_test, y_test) = tff.simulation.datasets.cifar100.load_data()
    x_train = x_train.reshape((-1, 32, 32, 3))
    x_test = x_test.reshape((-1, 32, 32, 3))
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return x_train, x_test


def get_input_spec():
    input_spec = collections.OrderedDict(
        x=tf.TensorSpec(shape=[None, 32*32], dtype=tf.float32, name='pixels'),
        y=tf.TensorSpec(shape=[None, None], dtype=tf.int64, name='label')
    )
    return input_spec


def server_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=1.0)


federated_train_data = setup_federated_data(cifar_train, sample_clients)
federated_train_data = [preprocess_dataset(
    federated_train_data) for x in federated_train_data]

iterative_process = tff.learning.build_federated_averaging_process(
    model_fn, client_optimizer_fn=client_optimizer_fn, server_optimizer_fn=server_optimizer_fn)
print(iterative_process.next.type_signature)
federated_eval = tff.learning.build_federated_evaluation(model_fn)
federated_algorithm = tff.templates.IterativeProcess(initialize_fn, next_fn)
server_state = federated_algorithm.initialize()

state, metrics = iterative_process.next(server_state, federated_train_data)
rs = evaluate(server_state, federated_train_data)
