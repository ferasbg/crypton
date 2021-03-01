import argparse
import logging
import os
import pickle
import random
import sys
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
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
from nn.network import Network
from PIL import Image
from tensorflow import keras

NUM_CLIENTS = 10
NUM_EPOCHS = 25
BATCH_SIZE = 32
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


class CryptoNetwork(Network):
    """
        Description: Deep Convolutional Neural Network With Secure Training and Testing
        Raises:
        Returns:
        References:
        Examples:

    """

    def __init__(self):
        super(CryptoNetwork, self).__init__()
        # get plaintext layers for network architecture, focus primarily on heavy dp and federated e.g. iterate on data processing to ImageDataGenerator and model.fit_generator() or model.fit()
        self.public_network = super().build_compile_model()
        self.crypto_network = self.build_compile_crypto_model()
        # perform encryption operations on the input images themselves before passing to network

    def build_compile_crypto_model(self):
        '''Build federated variant of convolutional network.'''
        # get binary matrix to pass np.shape of input image
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.reshape((-1, 32, 32, 3))
        x_test = x_test.reshape((-1, 32, 32, 3))
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        # specify input shape, loss function, plaintext network, and metrics
        input_spec = x_train[0].shape
        return tff.learning.from_keras_model(self.public_network, input_spec=input_spec, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])


    def main(self):
        raise NotImplementedError

if __name__ == '__main__':
    crypto_network = CryptoNetwork()
    crypto_network.build_compile_crypto_model()
    print(crypto_network.crypto_network)
