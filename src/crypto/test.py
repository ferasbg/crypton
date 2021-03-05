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

'''
# Features are intentionally sorted lexicographically by key for consistency
  # across datasets.
feature_dtypes = collections.OrderedDict(
  coarse_label=computation_types.TensorType(tf.int64),
  image=computation_types.TensorType(tf.uint8, shape=(32, 32, 3)),
  label=computation_types.TensorType(tf.int64))

'''

cifar_train, cifar_test = tff.simulation.datasets.cifar100.load_data() 

first_client_id = cifar_train.client_ids[0]
first_client_dataset = cifar_train.create_tf_dataset_for_client(
    first_client_id)
print(first_client_dataset.element_spec)
# OrderedDict([('coarse_label', TensorSpec(shape=(), dtype=tf.int64, name=None)), ('image', TensorSpec(shape=(32, 32, 3), dtype=tf.uint8, name=None)), ('label', TensorSpec(shape=(), dtype=tf.int64, name=None))])
assert cifar_train.element_type_structure == first_client_dataset.element_spec

# def preprocess(dataset):
#   def map_fn(input):
#     return collections.OrderedDict(
#         x=tf.reshape(input['pixels'], shape=(-1, 1024)),
#         y=tf.cast(tf.reshape(input['label'], shape=(-1, 1)), tf.int64),
#     )
  
#   return dataset.batch(BATCH_SIZE).map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).take(NUM_EXAMPLES_PER_CLIENT)

# # pass preprocess callable to client HDF5
# preprocessed_client_data = cifar_train.preprocess(preprocess) 

# first_client_dataset = preprocessed_client_data.create_tf_dataset_for_client(cifar_train.client_ids[0])

# print(first_client_dataset.element_spec)
