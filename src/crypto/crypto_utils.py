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

warnings.filterwarnings('ignore')

# preprocessed_client_data = cifar_test.preprocess(preprocess_fn) 
# preprocessed_and_shuffled = cifar_test.preprocess(preprocess_and_shuffle)
# selected_client_ids = preprocessed_and_shuffled.client_ids[:10] # 10 clients
# preprocessed_data_for_clients = [preprocessed_and_shuffled.create_tf_dataset_for_client(selected_client_ids[i]) for i in range(10)]
# client_ids = np.random.choice(cifar_train.client_ids, size=NUM_CLIENTS, replace=False)

'''
Algorithm for Federated Averaging

# for round_iter in range(NUM_ROUNDS):
    # for round n, client k is on epoch M iterating over batch size B
    # train local models for each client sequentially/concurrently, then take the average of all of the clients' gradients to then update the global model stored in the server
    # server_state, metrics = federated_algorithm.next(server_state) # next is an attribute, and not callable
    # print("Federated Metrics: {}".format(metrics))
    # evaluate(server_state)

    EXPECTED OUTPUT:
    round  2, metrics=OrderedDict([('broadcast', ()), ('aggregation', OrderedDict([('value_sum_process', ()), ('weight_sum_process', ())])), ('train', OrderedDict([('num_examples', 4860.0), ('loss', 2.941014), ('accuracy', 0.14218107)]))])
    round  3, metrics=OrderedDict([('broadcast', ()), ('aggregation', OrderedDict([('value_sum_process', ()), ('weight_sum_process', ())])), ('train', OrderedDict([('num_examples', 4860.0), ('loss', 2.9052832), ('accuracy', 0.14444445)]))])
    round  4, metrics=OrderedDict([('broadcast', ()), ('aggregation', OrderedDict([('value_sum_process', ()), ('weight_sum_process', ())])), ('train', OrderedDict([('num_examples', 4860.0), ('loss', 2.7491086), ('accuracy', 0.17962962)]))])
    round  5, metrics=OrderedDict([('broadcast', ()), ('aggregation', OrderedDict([('value_sum_process', ()), ('weight_sum_process', ())])), ('train', OrderedDict([('num_examples', 4860.0), ('loss', 2.5129666), ('accuracy', 0.19526748)]))])
    round  6, metrics=OrderedDict([('broadcast', ()), ('aggregation', OrderedDict([('value_sum_process', ()), ('weight_sum_process', ())])), ('train', OrderedDict([('num_examples', 4860.0), ('loss', 2.4175923), ('accuracy', 0.23600823)]))])
    round  7, metrics=OrderedDict([('broadcast', ()), ('aggregation', OrderedDict([('value_sum_process', ()), ('weight_sum_process', ())])), ('train', OrderedDict([('num_examples', 4860.0), ('loss', 2.4273515), ('accuracy', 0.24176955)]))])
    round  8, metrics=OrderedDict([('broadcast', ()), ('aggregation', OrderedDict([('value_sum_process', ()), ('weight_sum_process', ())])), ('train', OrderedDict([('num_examples', 4860.0), ('loss', 2.2426176), ('accuracy', 0.2802469)]))])
    round  9, metrics=OrderedDict([('broadcast', ()), ('aggregation', OrderedDict([('value_sum_process', ()), ('weight_sum_process', ())])), ('train', OrderedDict([('num_examples', 4860.0), ('loss', 2.1567981), ('accuracy', 0.295679)]))])
    round 10, metrics=OrderedDict([('broadcast', ()), ('aggregation', OrderedDict([('value_sum_process', ()), ('weight_sum_process', ())])), ('train', OrderedDict([('num_examples', 4860.0), ('loss', 2.1092515), ('accuracy', 0.30843621)]))])

'''

def model_fn():
    # build layers of public neural network to pass into tff constructor as plaintext model, do in function since it's static 
    model = Sequential()
    # feature layers
    model.add(tf.keras.Input(shape=(32,32,3)))
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
    crypto_network =  tff.learning.from_keras_model(model, input_spec=input_spec, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return crypto_network


def build_uncompiled_plaintext_keras_model():
  # build layers, do not compile model since federated evaluation will use optimizer through their own custom decorators
  # build layers of public neural network
  model = Sequential()
  # feature layers
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
    # stochastic gd has momentum, optimizer doesn't use momentum for weight regularization
  return model

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

@tff.federated_computation(federated_server_type, federated_dataset_type)
def next_fn(server_weights, federated_dataset):
  '''Compute client-to-server update and server-to-client update between nodes.
  Note, the server state stores the global model's gradients and its weights respectively given tff model state dict and optimizer_state e.g. vars of optimizer
  '''
  # broadcast server weights to client nodes for local model set
  server_weights_at_client = tff.federated_broadcast(server_weights)
  client_weights = tff.federated_map(client_update_fn, (federated_dataset, server_weights_at_client))

  # server averages client updates
  mean_client_weights = tff.federated_mean(client_weights)

  # server updates its model with average of the clients' updated gradients
  server_weights = tff.federated_map(server_update_fn, mean_client_weights)
  return server_weights
