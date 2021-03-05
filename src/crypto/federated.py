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
from tensorflow.python.keras.utils.data_utils import Sequence

from crypto_network import CryptoNetwork
from crypto_utils import model_fn

'''
pointers for constructors for federated eval / iterative_process Callable[], and some temporary reference

<tensorflow_federated.python.core.impl.computation.computation_impl.ComputationImpl object at 0x7fc33d4d9670>
<tensorflow_federated.python.core.templates.iterative_process.IterativeProcess object at 0x7fc33d76ce20>

import attr
@attr.s(eq=False, frozen=True)
class ServerState(object):
  """Structure for state on the server.
  Fields:
  -   `model`: A dictionary of model's trainable variables.
  -   `optimizer_state`: the list of variables of the optimizer.
  """
  model = attr.ib()
  optimizer_state = attr.ib()

  @classmethod
  def from_tff_result(cls, anon_tuple):
    return cls(
        model=tff.learning.framework.ModelWeights.from_tff_result(anon_tuple.model),
        optimizer_state=list(anon_tuple.optimizer_state))

example_dataset = emnist_test.create_tf_dataset_for_client(emnist_test.client_ids[client_id])

# note: what is stateful_delta_aggregate_fn ?
 @tff.federated_computation
  def server_init_tff():
    """Orchestration logic for server model initialization."""
    return tff.federated_value(server_init_tf(), tff.SERVER)

  return tff.templates.IterativeProcess(
      initialize_fn=server_init_tff, next_fn=run_one_round)

For example, if client A has been sampled m times at round n, and each time local model training has been run with q iterations, then the counter in client A state would be q*m at round n. On the server, we also aggregate the total number of iterations for all the clients sampled in this round.

- compute federated aggregation with federated gradient descent and secure aggregation algorithm for private, federated computation for neural networks training on client data locally, and sending average of gradient_updates to aggregator e.g. global model
- For each client k in K clients, given B = batch_size, E = epochs, n = learning_rate=0.001
- Rounds are iterations for each client iterating over each client node, and we want to sequentially iterate over each client node
- Use .model.function_name to train each model iteratively on local models that will update global model
- Use iterative_process.next(), federated_averaging_process(), and custom utils to setup clients nodes / server node
- implement federated averaging with federated_averaging_process() and iterative_process.next() and iterate_train_over_clients(), and update_client_nodes_to_global_aggregator()
- implement mpc-based secure aggregation (Bonawitz et. al, 2017), where the secrets are the model gradients that cannot reconstruct the original input data, since the average of the gradients are updated by the global model, where local model gradients are updated to the global model that is controlled by the aggregator that handles synchronous computation over K clients.
- attest to epistemic rigor given fed sim
- how does byzantine fault tolerance work for production-based federated learning?
# Albeit unrelated, but note that BatchNormalization() will destabilize local model instances because averaging over heterogeneous data and making averages over a non-linear distribution can create unstable effects on the neural network's performance locally, and then further distorting the shared global model whose weights are updated based on the updated state of the client's local model on-device or on-prem client-side.
# partition dataset (train) for each client so it acts as its own local data (private from other users during training, same global model used, update gradients to global model)
# let's just assume we are able to control the data that each client stores for models, that their status is available, their data isn't corrupted, and it's synchronous

# Research Question: how does variance of learning_rate AND low epoch_amount affect local and global model?
# Research Question: adding gradient norm clipping as an approach to regularize local client models given the adjusted learning_rate and adding learning_decay_rate and using l2_kernel_regularizer for weight regularization for plaintext keras network.

This recovers the original FedAvg algorithm in McMahan et al., 2017. More sophisticated federated averaging procedures may use different learning rates or server optimizers.

Remember the 4 elements of an FL algorithm?

A server-to-client broadcast step.
A local client update step.
A client-to-server upload step.
A server update step.

A key concept in TFF is "federated data", which refers to a collection of data items hosted across a group of devices in a distributed system (eg. client datasets, or the server model weights). We model the entire collection of data items across all devices as a single federated value

psuedocode for implementing state update between server that stores global model and clients that store local models:

def next_fn(server_weights, federated_dataset):
  # Broadcast the server weights to the clients.
  server_weights_at_client = broadcast(server_weights)

  # Each client computes their updated weights.
  client_weights = client_update(federated_dataset, server_weights_at_client)

  # The server averages these updates.
  mean_client_weights = mean(client_weights)

  # The server updates its model.
  server_weights = server_update(mean_client_weights)

  return server_weights

  We are dealing with non-iid data given K clients.

'''

# CONSTANTS
NUM_CLIENTS = 10
# MODEL TRAIN CONFIG
BATCH_SIZE = 20
NUM_EPOCHS = 5
# constant lr for SGD optimizer
CLIENT_LEARNING_RATE = 0.02
SERVER_LEARNING_RATE = 1.0
NUM_ROUNDS = 5
CLIENTS_PER_ROUND = 2
CLIENT_EPOCHS_PER_ROUND = 1
# helper constants for data preprocessing given tf.data.Dataset to tff native transformations
SHUFFLE_BUFFER = 100 # shuffling
PREFETCH_BUFFER = 10 # data to prefetch in cache for training

# federated cifar-100 dataset: 500 train clients, 100 test clients
cifar_train, cifar_test = tff.simulation.datasets.cifar100.load_data()

'''
"""Flatten a batch `pixels` and return the features as an `OrderedDict`.

===> Datset.from_tensors(train_image_set, train_label_set)
===> return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
===> BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

There's a difference between passing callable as param, calling member function of object/callable, nesting functions for recursion or iteration, referencing to Callable[] like `obj.next` instead of `obj.member_function()`

References:
    - https://cs230.stanford.edu/blog/datapipeline/
    - https://www.tensorflow.org/api_docs/python/tf/data/Dataset
'''


def preprocess_dataset(dataset):
  def map_fn(element):
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 784]),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.batch(BATCH_SIZE).map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).take(1000)

# pass callable to preprocess
preprocessed_client_data = cifar_test.preprocess(preprocess_dataset) 
client_dataset = []

def generate_clients():
  for i in range(10):
    client_test_set = preprocessed_client_data.create_tf_dataset_for_client(cifar_test.client_ids[i])
    print(client_test_set.element_spec)
    assert cifar_test.element_type_structure == client_test_set.element_spec
    client_dataset.append(client_test_set)


def preprocess_and_shuffle(dataset):
  preprocessed = preprocess_dataset(dataset)
  return preprocessed.shuffle(SHUFFLE_BUFFER)

preprocessed_and_shuffled = cifar_test.preprocess(preprocess_and_shuffle)
selected_client_ids = preprocessed_and_shuffled.client_ids[:10] # 10 clients

# setup preprocessed data for each client
preprocessed_data_for_clients = [preprocessed_and_shuffled.create_tf_dataset_for_client(selected_client_ids[i]) for i in range(10)]

# setup clients with cifar-100 train
client_ids = np.random.choice(cifar_train.client_ids, size=NUM_CLIENTS, replace=False)


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
  # federated_value() was deprecated
  # note that fn is a non-arg tff computation
  return tff.federated_value(server_init(), tff.SERVER)

# define types to properly decorate tff functions
dummy_model = model_fn()
model_weights_type = server_init.type_signature.result
tf_dataset_type = tff.SequenceType(dummy_model.input_spec)
federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

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

# setup IterativeProcess and federated_eval for Federated Evaluation and Federated Averaging
iterative_process = tff.learning.build_federated_averaging_process(model_fn, CryptoNetwork.client_optimizer_fn, CryptoNetwork.server_optimizer_fn)
print(federated_algorithm.next.type_signature)

# note that federated_eval returns the federated_output_computation, which computes federated aggregation of the model's local_inputs e.g. compute function federated network over its input for client node
federated_eval = tff.learning.build_federated_evaluation(model_fn, use_experimental_simulation_loop=False) #  takes a model function and returns a single federated computation for federated evaluation of models, since evaluation is not stateful.
print(str(federated_eval.type_signature))


# accept server model and client data given client models, and then return an updated server model with defined with federated averaging process
federated_algorithm = tff.templates.IterativeProcess(initialize_fn, next_fn)
# define federated eval on given server_state given weights and gradients and model architecture
central_cifar_test = cifar_test.create_tf_dataset_from_all_clients().take(1000) # 1000 images for test

# custom eval given server_state
def evaluate(server_state):
  # use plaintext model
  network = build_uncompiled_plaintext_keras_model()
  network.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  network.set_weights(server_state) # vectorized state of network in server
  network.evaluate(central_cifar_test) # pass data to keras model

# define ServerState and ClientState, initialize state of tff computation
server_state = federated_algorithm.initialize()

orchestrate_federated_accepting_ids = tff.simulation.compose_dataset_computation_with_iterative_process(preprocessed_and_shuffled.dataset_computation, iterative_process)

for round in range(NUM_ROUNDS):
  server_state, metrics = orchestrate_federated_accepting_ids.next(server_state, selected_client_ids)
  evaluate(server_state)  

# RUN ITERATIVE PROCESS TO COMPUTE FEDERATED EVALUATION WITH FEDERATED AVERAGING

# assign server model state and its metrics to the abstract tff computation to compute tff state transition
# for round_iter in range(NUM_ROUNDS):
    # for round n, client k is on epoch M iterating over batch size B
    # train local models for each client sequentially/concurrently, then take the average of all of the clients' gradients to then update the global model stored in the server
    # server_state, metrics = federated_algorithm.next(server_state) # next is an attribute, and not callable
    # print("Federated Metrics: {}".format(metrics))
    # evaluate(server_state)



'''
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


