import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import random
import pickle
import tensorflow_federated as tff
from crypto_network import CryptoNetwork
from crypto_utils import model_fn

'''
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
# Question: how does variance of learning_rate AND low epoch_amount affect local and global model?
    
'''

# CONSTANTS
BATCH_SIZE = 20
EPOCHS = 1
# constant lr for optimizer
CLIENT_LEARNING_RATE = 0.02
SERVER_LEARNING_RATE = 1.0 
# let's just assume we are able to control the data that each client stores for models, that their status is available, their data isn't corrupted, and it's synchronous
NUM_ROUNDS = 5
CLIENTS_PER_ROUND = 2
CLIENT_EPOCHS_PER_ROUND = 1
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


# plaintext data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.reshape((-1, 32, 32, 3))
x_test = x_test.reshape((-1, 32, 32, 3))
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# federated data
cifar_train, cifar_test = tff.simulation.datasets.cifar100.load_data()
iterative_process = tff.learning.build_federated_averaging_process(model_fn, CryptoNetwork.client_optimizer_fn, CryptoNetwork.server_optimizer_fn)
federated_eval = tff.learning.build_federated_evaluation(model_fn, use_experimental_simulation_loop=False) #  takes a model function and returns a single federated computation for federated evaluation of models, since evaluation is not stateful.

# 500 train clients, 100 test clients
print(federated_eval)
print(iterative_process)


