import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import random
import pickle
import tensorflow_federated as tff
from src.nn.network import Network

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
        sample_batch =x_train[0]
        return tff.learning.from_keras_model(self.public_network, sample_batch)


    def main(self):
        raise NotImplementedError

