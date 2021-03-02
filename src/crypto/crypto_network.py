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

from crypto_utils import model_fn

warnings.filterwarnings('ignore')

NUM_CLIENTS = 10
NUM_EPOCHS = 25
BATCH_SIZE = 32
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


class Network():
    classification_state = False
    dataset_labels = ['airplane', 'automobile', 'bird',
                      'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # list of ints that represent the labels given self.dataset_labels so basically index the list of dataset_labels given y_train[i] with Network.dataset_labels[i]

    def __init__(self):

        # labels
        self.num_classes = 20
        self.epochs = 1000
        self.batch_size = 64  # maybe 128
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.kernel_size = 3
        self.stride = 2
        self.padding = 1
        self.dilation = 1
        self.epochs = 500
        self.input_channels = 3
        self.output_channels = 64  # number of channels produced by convolution
        self.image_size = [32, 32]
        self.bias = False
        # stabilize convergence to local minima for gradient descent
        self.weight_decay_regularization = 0.003 # us kernel_regularizer in Conv2d as iteration
        self.momentum = 0.05  # gradient descent convergence optimizer
        self.model = self.build_compile_model()

    def build_compile_model(self):
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
        optimizer = Adam(learning_rate=0.001)
        # model.compile(loss='categorical_crossentropy',
        #               optimizer=optimizer, metrics=['accuracy'])
        return model

    def train(self):
        '''
        Train network on nominal multi-label classification problem. Make sure to allocate images for each test variation e.g. mpc_network, certified_mpc_network, certified_nominal_network

        # partitioning dataset for different tests.
        x_val = x_train[-10000:]
        y_val = y_train[-10000:]
        x_train = x_train[:-10000]
        y_train = y_train[:-10000]

        # if 10 clients and total of 50000 images, assume 1000 images per client

        Add dataset, and dataset_labels as parameter (overload this method or make re-write)

        '''
        # x_train stores all of the train_images and y_train stores all the respective categories of each image, in the same order.
        # get cifar10-data first, and assign data and categorical labels as such
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        x_train = x_train.reshape((-1, 32, 32, 3))
        x_test = x_test.reshape((-1, 32, 32, 3))

        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        generator = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False)

        history = self.model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))
        self.model.evaluate(x=x_test, y=y_test, verbose=0)

        self.model.save_weights('network.h5')
        print(history)
        

    @staticmethod
    def getClassificationState():
        return Network.classification_state

class CryptoNetwork(Network):
    """
        Deep Convolutional Neural Network With Federated Computation 

        Note that in real-world production scenario, we would have to analyze the states of our clients such that local training is possible since on-prem only works when their devices are on. Also note that data in the real-world is always messy and not clean as the datasets being used for this process.

        "There is a fixed set of K clients, each with a fixed local dataset. At the beginning of each round, a random fraction C of clients is selected, and the server sends the current global algorithm state to each of these clients (e.g., the current model parameters). We only select a fraction of clients for efficiency, as our experiments show diminishing returns for adding more clients beyond a certain point. Each selected client then performs local computation based on the global state and its local dataset, and sends an update to the server. The server then applies these updates to its global state, and the process repeats." (McMahan et. al)

        "The key consequence of this is that federated computations, by design, are expressed in a manner that is oblivious to the exact set of participants; all processing is expressed as aggregate operations on an abstract group of anonymous clients, and that group might vary from one round of training to another. The actual binding of the computation to the concrete participants, and thus to the concrete data they feed into the computation, is thus modeled outside of the computation itself."

        Raises:
        Returns:
        References:
        Examples:

    """

    def __init__(self):
        '''Build federated variant of convolutional network.'''
        super(CryptoNetwork, self).__init__()
        # get plaintext layers for network architecture, focus primarily on heavy dp and federated e.g. iterate on data processing to ImageDataGenerator and model.fit_generator() or model.fit()
        self.public_network = super().build_compile_model()
        self.input_spec = collections.OrderedDict(
            x=tf.TensorSpec(shape=[None, None], dtype=tf.float32),
            y=tf.TensorSpec(shape=[None, 1], dtype=tf.int64)  
        )
        # tff wants new tff network created upon instantiation or invocation of method call
        self.crypto_network =  tff.learning.from_keras_model(self.public_network, input_spec=self.input_spec, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])

    def getTrainingData(self):
        (x_train, y_train), (x_test, y_test) = tff.simulation.datasets.cifar100.load_data()
        x_train = x_train.reshape((-1, 32, 32, 3))
        x_test = x_test.reshape((-1, 32, 32, 3))
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        return x_train, x_test 

    @staticmethod
    def client_optimizer_fn():
        return tf.keras.optimizers.SGD(learning_rate=0.02)

    @staticmethod
    def server_optimizer_fn():
        return tf.keras.optimizers.SGD(learning_rate=1.0)


if __name__ == '__main__':
    # setup crypto_network, federated_dataset, federated_clients, setup federated_eval() 
    crypto_network = CryptoNetwork()
    print("Trainable variables: ", crypto_network.crypto_network.trainable_variables)
    cifar_train, cifar_test = tff.simulation.datasets.cifar100.load_data()
    
    federated_eval = tff.learning.build_federated_evaluation(model_fn, use_experimental_simulation_loop=False) #  takes a model function and returns a single federated computation for federated evaluation of models, since evaluation is not stateful.
    iterative_process = tff.learning.build_federated_averaging_process(model_fn, CryptoNetwork.client_optimizer_fn, CryptoNetwork.server_optimizer_fn)
    print(federated_eval)
        
    # compute federated averaging e.g. computing over K clients.        
    # Albeit unrelated, but note that BatchNormalization() will destabilize local model instances because averaging over heterogeneous data and making averages over a non-linear distribution can create unstable effects on the neural network's performance locally, and then further distorting the shared global model whose weights are updated based on the updated state of the client's local model on-device or on-prem client-side. 
    # partition dataset (train) for each client so it acts as its own local data (private from other users during training, same global model used, update gradients to global model)        
    # Question: how does variance of learning_rate AND low epoch_amount affect local and global model?
    