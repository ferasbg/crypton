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
from PIL import Image
from tensorflow import keras

from crypto_network import CryptoNetwork
from crypto_utils import model_fn, build_uncompiled_plaintext_keras_model, server_init, server_update, client_update_fn, server_update_fn, next_fn, initialize_fn
'''setup IterativeProcess and federated_eval for Federated Evaluation and Federated Averaging'''

# CONSTANTS
NUM_CLIENTS = 10
# MODEL TRAIN CONFIG
BATCH_SIZE = 20
NUM_EPOCHS = 10
# constant lr for SGD optimizer
CLIENT_LEARNING_RATE = 0.02
SERVER_LEARNING_RATE = 1.0
NUM_ROUNDS = 5
CLIENTS_PER_ROUND = 2
NUM_EXAMPLES_PER_CLIENT = 500
CIFAR_SHAPE = (32,32,3)
TOTAL_FEATURE_SIZE = 32 * 32 * 3
CLIENT_EPOCHS_PER_ROUND = 1
SHUFFLE_BUFFER = 100 # shuffling
PREFETCH_BUFFER = 10 # data to prefetch in cache for training
client_dataset = []
cifar_train, cifar_test = tff.simulation.datasets.cifar100.load_data()
sample_clients = cifar_train.client_ids[0:NUM_CLIENTS]

def preprocess(dataset):
    def element_fn(element):
        # vectorize each image for all images in client's dataset
        return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 784]),
        y=tf.reshape(element['label'], [-1, 1]))
    
    return dataset.repeat(NUM_EPOCHS).map(element_fn).batch(BATCH_SIZE)

def make_federated_data(client_data, client_ids):
  # create dataset for client, then iterate pre-processing for each image in client dataset for all clients
    return [preprocess(client_data.create_tf_dataset_for_client(x) for x in client_ids)]

# pass entire cifar train dataset and partition given clients
federated_train_data = make_federated_data(cifar_train, sample_clients)
# iterative_process = tff.learning.build_federated_averaging_process(model_fn, CryptoNetwork.client_optimizer_fn, CryptoNetwork.server_optimizer_fn)
iterative_process = tff.learning.build_federated_averaging_process(model_fn, client_optimizer_fn=CryptoNetwork.client_optimizer_fn, server_optimizer_fn=CryptoNetwork.server_optimizer_fn)
# note that federated_eval returns the federated_output_computation, which computes federated aggregation of the model's local_inputs e.g. compute function federated network over its input for client node
federated_eval = tff.learning.build_federated_evaluation(model_fn, use_experimental_simulation_loop=False) #  takes a model function and returns a single federated computation for federated evaluation of models, since evaluation is not stateful.
print(str(federated_eval.type_signature))
# accept server model and client data given client models, and then return an updated server model with defined with federated averaging process
federated_algorithm = tff.templates.IterativeProcess(initialize_fn, next_fn)
print(federated_algorithm.next.type_signature)
# define ServerState and ClientState, initialize state of tff computation
server_state = federated_algorithm.initialize()

# next isn't callable, yet each code sample passes args 
print(iterative_process.next.type_signature)

# eval server_state
def evaluate(server_state):
  # use plaintext model
  network = build_uncompiled_plaintext_keras_model()
  network.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  network.set_weights(server_state) # vectorized state of network in server
  network.evaluate(federated_train_data) # pass data to keras model


# for round in range(NUM_ROUNDS):
#   server_state = federated_algorithm.next(server_state, federated_train_data) # this isn't callable, so why do ppl do this method? or was it psuedocode
#   evaluate(server_state)  
