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

'''
Crypto stores logics for differential privacy and federated learning techniques. If secrets are computed as subsets of the entire composition function (network) f(x), then we must apply this to the context of a federated setting.

'''

global_node = CryptoNetwork().model

# clients
client_1 = CryptoNetwork().model
client_2 = CryptoNetwork().model
client_3 = CryptoNetwork().model
client_4 = CryptoNetwork().model
client_5 = CryptoNetwork().model
client_6 = CryptoNetwork().model
client_7 = CryptoNetwork().model
client_8 = CryptoNetwork().model
client_9 = CryptoNetwork().model
client_10 = CryptoNetwork().model

clients = []

clients.append(client_1)
clients.append(client_2)
clients.append(client_3)
clients.append(client_4)
clients.append(client_5)
clients.append(client_6)
clients.append(client_7)
clients.append(client_8)
clients.append(client_9)
clients.append(client_10)

# for each client k in K clients, given B = batch_size, E = epochs, n = learning_rate=0.001
# rounds are iterations, and we want to sequentially iterate over each client node
# use .model.function_name to train each model iteratively on local models that will update global model
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.reshape((-1, 32, 32, 3))
x_test = x_test.reshape((-1, 32, 32, 3))

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# each client gets 1000 images
BATCH_SIZE = 32
EPOCHS = 25

## CLIENT DATASET GENERATION

# add data and labels for each client
client_train_dataset_1 = x_train[-1250:]
client_train_labels_1 = y_train[-1250:]
# remove train
x_train = x_train[:-1250]
y_train = y_train[:-1250]
client_validation_data_1 = x_test[-500:]
client_validation_labels_1 = y_test[-500:]
# remove validation
x_test = x_test[:-500]
y_test = y_test[:-500]



# add data and labels for each client
client_train_dataset_2 = x_train[-1250:]
client_train_labels_2 = y_train[-1250:]
# remove train
x_train = x_train[:-1250]
y_train = y_train[:-1250]
client_validation_data_2 = x_test[-500:]
client_validation_labels_2 = y_test[-500:]
# remove validation
x_test = x_test[:-500]
y_test = y_test[:-500]


# add data and labels for each client
client_train_dataset_3 = x_train[-1250:]
client_train_labels_3 = y_train[-1250:]
# remove train
x_train = x_train[:-1250]
y_train = y_train[:-1250]
client_validation_data_3 = x_test[-500:]
client_validation_labels_3 = y_test[-500:]
# remove validation
x_test = x_test[:-500]
y_test = y_test[:-500]




# add data and labels for each client
client_train_dataset_4 = x_train[-1250:]
client_train_labels_4 = y_train[-1250:]
# remove train
x_train = x_train[:-1250]
y_train = y_train[:-1250]
client_validation_data_4 = x_test[-500:]
client_validation_labels_4 = y_test[-500:]
# remove validation
x_test = x_test[:-500]
y_test = y_test[:-500]



# add data and labels for each client
client_train_dataset_5 = x_train[-1250:]
client_train_labels_5 = y_train[-1250:]
# remove train
x_train = x_train[:-1250]
y_train = y_train[:-1250]
client_validation_data_5 = x_test[-500:]
client_validation_labels_5 = y_test[-500:]
# remove validation
x_test = x_test[:-500]
y_test = y_test[:-500]



# add data and labels for each client
client_train_dataset_6 = x_train[-1250:]
client_train_labels_6 = y_train[-1250:]
# remove train
x_train = x_train[:-1250]
y_train = y_train[:-1250]
client_validation_data_6 = x_test[-500:]
client_validation_labels_6 = y_test[-500:]
# remove validation
x_test = x_test[:-500]
y_test = y_test[:-500]



# add data and labels for each client
client_train_dataset_7 = x_train[-1250:]
client_train_labels_7 = y_train[-1250:]
# remove train
x_train = x_train[:-1250]
y_train = y_train[:-1250]
client_validation_data_7 = x_test[-500:]
client_validation_labels_7 = y_test[-500:]
# remove validation
x_test = x_test[:-500]
y_test = y_test[:-500]




# add data and labels for each client
client_train_dataset_8 = x_train[-1250:]
client_train_labels_8 = y_train[-1250:]
# remove train
x_train = x_train[:-1250]
y_train = y_train[:-1250]
client_validation_data_8 = x_test[-500:]
client_validation_labels_8 = y_test[-500:]
# remove validation
x_test = x_test[:-500]
y_test = y_test[:-500]




# add data and labels for each client
client_train_dataset_9 = x_train[-1250:]
client_train_labels_9 = y_train[-1250:]
# remove train
x_train = x_train[:-1250]
y_train = y_train[:-1250]
client_validation_data_9 = x_test[-500:]
client_validation_labels_9 = y_test[-500:]
# remove validation
x_test = x_test[:-500]
y_test = y_test[:-500]




# add data and labels for each client
client_train_dataset_10 = x_train[-1250:]
client_train_labels_10 = y_train[-1250:]
# remove train
x_train = x_train[:-1250]
y_train = y_train[:-1250]
client_validation_data_10 = x_test[-500:]
client_validation_labels_10 = y_test[-500:]
# remove validation
x_test = x_test[:-500]
y_test = y_test[:-500]




client_train_datasets = []
# add client train sets to list
client_train_datasets.append(client_train_dataset_1)
client_train_datasets.append(client_train_dataset_2)
client_train_datasets.append(client_train_dataset_3)
client_train_datasets.append(client_train_dataset_4)
client_train_datasets.append(client_train_dataset_5)
client_train_datasets.append(client_train_dataset_6)
client_train_datasets.append(client_train_dataset_7)
client_train_datasets.append(client_train_dataset_8)
client_train_datasets.append(client_train_dataset_9)
client_train_datasets.append(client_train_dataset_10)


client_train_labels = []
client_train_labels.append(client_train_labels_1)
client_train_labels.append(client_train_labels_2)
client_train_labels.append(client_train_labels_3)
client_train_labels.append(client_train_labels_4)
client_train_labels.append(client_train_labels_5)
client_train_labels.append(client_train_labels_6)
client_train_labels.append(client_train_labels_7)
client_train_labels.append(client_train_labels_8)
client_train_labels.append(client_train_labels_9)
client_train_labels.append(client_train_labels_10)


client_validation_data = []
client_validation_data.append(client_validation_data_1)
client_validation_data.append(client_validation_data_2)
client_validation_data.append(client_validation_data_3)
client_validation_data.append(client_validation_data_4)
client_validation_data.append(client_validation_data_5)
client_validation_data.append(client_validation_data_6)
client_validation_data.append(client_validation_data_7)
client_validation_data.append(client_validation_data_8)
client_validation_data.append(client_validation_data_9)
client_validation_data.append(client_validation_data_10)




client_validation_labels = []
client_validation_labels.append(client_validation_labels_1)
client_validation_labels.append(client_validation_labels_2)
client_validation_labels.append(client_validation_labels_3)
client_validation_labels.append(client_validation_labels_4)
client_validation_labels.append(client_validation_labels_5)
client_validation_labels.append(client_validation_labels_6)
client_validation_labels.append(client_validation_labels_7)
client_validation_labels.append(client_validation_labels_8)
client_validation_labels.append(client_validation_labels_9)
client_validation_labels.append(client_validation_labels_10)

i = 0

for round in range(10):
    for client in clients:
        client.federated_train(BATCH_SIZE, EPOCHS, client_train_data=client_train_datasets[i], client_train_labels=client_train_labels[i], client_validation_data=client_validation_data[i], client_validation_labels=client_validation_labels[i])
        i+=1
        if (i == len(clients)):
            break

# get average of all the weights from all of the clients to update global model


