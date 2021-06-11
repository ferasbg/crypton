import argparse
import collections
import logging
import os
import pickle
import random
import sys
import time
import warnings

import flwr
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_privacy as tfp
import tensorflow_probability as tfpb
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
from model import DefenseNetwork, Network
from PIL import Image
from tensorflow import keras

from federated.crypto_utils import *
from federated.tff.tff_utils import *

warnings.filterwarnings('ignore')

class CryptoNetwork(Network):
    """
    Description: Deep Convolutional Neural Network for Federated Learning 

    """

    def __init__(self):
        super(CryptoNetwork, self).__init__()
        self.public_network = super().build_compile_model()
        self.uncompiled_public_network = super().build_uncompiled_model()
        self.model_weights = self.public_network.get_weights() 
        self.flower_network = []
        self.tff_network =  tff.learning.from_keras_model(self.public_network, input_spec=get_input_spec(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

class Client(CryptoNetwork):
    '''
    The `Client` model can apply defenses to the self.public_network variable inherited from CryptoNetwork before wrapping it into a federated client model.

    '''
    def __init__(self, defense_state):
        # two functions that modify the layers of the crypto network
        # client <  crypto network < network
        super(Client, self).__init__()
        # copy of the public network
        self.public_network = super().public_network
        # client network without gaussian or random transformations
        self.client_network = super().flower_network
        if (defense_state == True):
            self.defense_network = super().public_network.apply_defenses()

    def apply_defenses(self):
        # this function exists so we can write more defenses to modify both our model and the data passed to it
        # apply these transformations to modify the uncompiled plaintext model that will be passed into the crypto_network object inherited from CryptoNetwork. 
        self.apply_gaussian_layer()

    def apply_gaussian_layer(self):
        # the goal is to prevent adversarial over-fitting during this form of "augmentation"
        client_model = Sequential()
        gaussian_layer = tf.keras.layers.GaussianNoise()
        client_model.add(gaussian_layer)
        client_model.add(Conv2D(32, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        client_model.add(BatchNormalization())
        client_model.add(Dropout(0.3))
        client_model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        client_model.add(MaxPool2D((2, 2)))
        client_model.add(Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        client_model.add(Conv2D(128, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        client_model.add(MaxPool2D((2, 2)))
        client_model.add(Conv2D(128, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        client_model.add(Conv2D(256, (3, 3), activation='relu',
                         kernel_initializer='he_uniform', padding='same'))
        client_model.add(MaxPool2D((2, 2)))
        client_model.add(Flatten())
        client_model.add(Dense(128, activation='relu',
                        kernel_initializer='he_uniform'))
        client_model.add(Dense(10, activation='softmax'))
        # uncompiled for tff wrapper that is passed in super().crypto_network
        self.public_network = client_model

class TrustedAggregator(CryptoNetwork):
    # server model
    def __init__(self):
        super(TrustedAggregator, self).__init__()
