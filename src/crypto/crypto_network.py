import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import random
import pickle
import tf_encrypted as tfe
# import tf_federated as tff

from nn.network import Network


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
        self.crypto_network = super().build_compile_model()
        # perform encryption operations on the input images themselves before passing to network 
        # encrypt layers with self.MPCConjugateLayer = MPC.encryptPublicConjugateLayer(Network.PublicConjugateLayer)

    def main(self):
        raise NotImplementedError

