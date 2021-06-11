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
import tensorflow_privacy as tfp
import tensorflow_probability as tfpb 

from model import Network
from crypto.crypto_utils import model_fn, build_uncompiled_plaintext_keras_model, server_init, server_update, client_update_fn, server_update_fn, next_fn, initialize_fn
warnings.filterwarnings('ignore')

class CryptoNetwork(Network):
    """
    Description: Deep Convolutional Neural Network With Federated Computation with wrapped tff.Computation. 

    Context:
        - It's true that in reality we'd use few-shot or unsupervised models and that'd there be corrupted devices, faults, and so on with both the data and the clients' devices.
        - There is a fixed set of K clients, each with a fixed local dataset. At the beginning of each round, a random fraction C of clients is selected, and the server sends the current global algorithm state to each of these clients (e.g., the current model parameters; client_receiver = server.send_global_state()). Each selected client then performs local computation based on the global state and its local dataset, and then at the end of the round the updated gradients all of the clients are averaged and sent to the server model. The goal is for the trusted aggregator model to generalize well given the distributional variance in the data per client (or the expected variance : different image types). The server then applies these updates to its global state, and the process repeats. 
    """

    def __init__(self):
        '''Creates convolutional network that is wrapped with Tensorflow Federated.'''
        super(CryptoNetwork, self).__init__()
        # get plaintext layers for network architecture, focus primarily on heavy dp and federated e.g. iterate on data processing to ImageDataGenerator and model.fit_generator() or model.fit()
        self.public_network = super().build_compile_model()
        self.input_spec = collections.OrderedDict(
            x=tf.TensorSpec(shape=[None, 32*32], dtype=tf.float32, name='pixels'),
            y=tf.TensorSpec(shape=[None, None], dtype=tf.int64, name='label')  
        )
        # tff wants new tff network created upon instantiation or invocation of method call
        self.crypto_network =  tff.learning.from_keras_model(self.public_network, input_spec=self.input_spec, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        self.model_weights = self.public_network.get_weights() # not sure how this'd be secure during fed avg. (can we store an array of client model weights?)

    def get_training_data(self):
        (x_train, y_train), (x_test, y_test) = tff.simulation.datasets.cifar100.load_data()
        x_train = x_train.reshape((-1, 32, 32, 3))
        x_test = x_test.reshape((-1, 32, 32, 3))
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        return x_train, x_test 

    @staticmethod
    def client_optimizer_fn():
        # client optimizer_fn updates local client model while server_optimizer_fn applies the averaged update to the global model in the server
        # implementation: client_optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.02)
        return tf.keras.optimizers.SGD(learning_rate=0.02)

    @staticmethod
    def server_optimizer_fn():
        return tf.keras.optimizers.SGD(learning_rate=1.0)

    @staticmethod
    def make_federated_eval():
        # takes a model function and returns a single federated computation for federated evaluation of models, since evaluation is not stateful.
        federated_eval = tff.learning.build_federated_evaluation(model_fn) 
        return federated_eval

    @staticmethod
    def evaluate(server_state, federated_dataset):
        network = build_uncompiled_plaintext_keras_model()
        network.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        network.set_weights(server_state) # vectorized state of network in server
        network.evaluate(federated_dataset) # pass data to keras model
