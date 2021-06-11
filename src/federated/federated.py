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

from crypto.crypto_utils import model_fn, build_uncompiled_plaintext_keras_model, server_init, server_update, client_update_fn, server_update_fn, next_fn, initialize_fn

def fed_avg():
  pass

def fed_eval():
  pass

'''
Purpose: Compute Federated Evaluation and Federated Averaging.
Description:
  - We can assess given IID data, so note this statistical significance in terms of its effects during forwardpropagation and gradient accumulation.
  - No clients share any data samples, so it is a true partition of CIFAR-100. The train clients have string client IDs in the range [0-499], while the test clients have string client IDs in the range [0-99]. The train clients form a true partition of the CIFAR-100 training split, while the test clients form a true partition of the CIFAR-100 testing split. The data partitioning is done using a hierarchical Latent Dirichlet Allocation (LDA) process, referred to as the Pachinko Allocation Method (PAM). This method uses a two-stage LDA process, where each client has an associated multinomial distribution over the coarse labels of CIFAR-100, and a coarse-to-fine label multinomial distribution for that coarse label over the labels under that coarse label. The coarse label multinomial is drawn from a symmetric Dirichlet with parameter 0.1, and each coarse-to-fine multinomial distribution is drawn from a symmetric Dirichlet with parameter 10. Each client has 100 samples. To generate a sample for the client, we first select a coarse label by drawing from the coarse label multinomial distribution, and then draw a fine label using the coarse-to-fine multinomial distribution. We then randomly draw a sample from CIFAR-100 with that label (without replacement). If this exhausts the set of samples with this label, we remove the label from the coarse-to-fine multinomial and renormalize the multinomial distribution.
  - Each time the `next` method is called, the server model is broadcast to each client using a broadcast function. For each client, one epoch of local training is performed via the `tf.keras.optimizers.Optimizer.apply_gradients` method of the client optimizer. Each client computes the difference between the client model after training and the initial broadcast model. These model deltas are then aggregated at the server using some aggregation function. The aggregate model delta is applied at the server by using the `tf.keras.optimizers.Optimizer.apply_gradients` method of the server optimizer. Note: the default server optimizer function is `tf.keras.optimizers.SGD` with a learning rate of 1.0, which corresponds to adding the model delta to the current server model. This recovers the original FedAvg algorithm in [McMahan et al., 2017](https://arxiv.org/abs/1602.05629). More sophisticated federated averaging procedures may use different learning rates or server optimizer.
  - Realistically, there'd be some form of a queue algorithm that would handle checking for active and available client nodes that also have data that adhere to constraints relevant to training the model on-prem anyway.
  - Features are intentionally sorted lexicographically by key for consistency across datasets.
    feature_dtypes = collections.OrderedDict(
      coarse_label=computation_types.TensorType(tf.int64),
      image=computation_types.TensorType(tf.uint8, shape=(32, 32, 3)),
      label=computation_types.TensorType(tf.int64))
  - Pushing dataset construction and preprocessing to the clients avoids bottlenecks in serialization, and significantly increases performance with hundreds-to-thousands of clients.
  - Evaluating types for Tensorflow Federated to validate the nested method calls and data objects
    # print(str("FEDERATED EVAL TYPE SIGNATURE", federated_eval.type_signature))
    # print(iterative_process.next.type_signature)
    # print(federated_algorithm.next.type_signature)
    # print(len(cifar_train.client_ids)) # 500 client ids for cifar-100
    # print(cifar_train.element_type_structure) # (32,32,3)

  - Federated Averaging Algorithm:
    1. initialize algo, get initial server state, which stores necessary type and information to perform computation
    2. given functional types, state includes optimizer state its using e.g. tf.keras.optimizers.SGD(lr=1e-3, momentum=0.9) as well as the model params, passed as args
    3. execute algorithm in terms of rounds, where for each round, a new server state will be returned as the result of each client training the model on its data.
      a. server broadcast the model to all the participating client nodes
      b. each client performs work based on the model and its own data
      c. server aggregates all the models to produce a server state which contains a new model, taking the average of the updated gradients of all the participant client nodes

References:
  Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629

'''

NUM_CLIENTS = 10
BATCH_SIZE = 20
NUM_EPOCHS = 10
CLIENT_LEARNING_RATE = 0.02
SERVER_LEARNING_RATE = 1.0
NUM_ROUNDS = 5
CLIENTS_PER_ROUND = 2
NUM_EXAMPLES_PER_CLIENT = 500
CIFAR_SHAPE = (32,32,3)
TOTAL_FEATURE_SIZE = 32 * 32 * 3
CLIENT_EPOCHS_PER_ROUND = 1
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
client_dataset = []
cifar_train, cifar_test = tff.simulation.datasets.cifar100.load_data()
sample_clients = cifar_train.client_ids[0:NUM_CLIENTS]

def preprocess_dataset(dataset):
  def map_fn(element):
    return collections.OrderedDict(
        x=tf.cast(tf.reshape(element['image'], shape=(-1, 32, 32, 3)), tf.float32),
        y=tf.cast(tf.reshape(element['label'], shape=(-1, 1)), tf.int32),
    )

  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).map(map_fn).prefetch(PREFETCH_BUFFER)

def setup_federated_data(client_data, client_ids, federated_train_data=[]):
  for x in client_ids:
    client_node_data = preprocess_dataset(client_data.create_tf_dataset_for_client(x))
    federated_train_data.append(client_node_data)
  return federated_train_data

def make_federated_data(client_data, client_ids):
  return [preprocess_dataset(client_data.create_tf_dataset_for_client(x) for x in client_ids)]

def federated_train_generator():
  return [(federated_train_data[x] for x in federated_train_data)]

# i did not follow the by-the-book approach of using make_federated_data such that it returns the iterable because it was having problems with understanding the generator param passed in the .next() method. Instead I just stored it in a list, and need to convert it into an iterable through some function of some sort when accessing the data.
federated_train_data = setup_federated_data(cifar_train, sample_clients)
federated_train_data = [preprocess_dataset(federated_train_data) for x in federated_train_data]

iterative_process = tff.learning.build_federated_averaging_process(model_fn, client_optimizer_fn=client_optimizer_fn, server_optimizer_fn=server_optimizer_fn)
federated_eval = tff.learning.build_federated_evaluation(model_fn)
federated_algorithm = tff.templates.IterativeProcess(initialize_fn, next_fn)
server_state = federated_algorithm.initialize()

# state, metrics = iterative_process.next(server_state, federated_train_data)
# CryptoNetwork.evaluate(server_state, federated_train_data)
