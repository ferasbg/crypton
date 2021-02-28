import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import random
import pickle
import tf_encrypted as tfe

from nn.network import Network


class MPCNetwork(Network):
    """
        Description: Deep Convolutional Neural Network With Secure Training and Testing
        Raises:
            - Error if NetworkStateEncryption=NULL, if Sanity Checks Fail, if Reconstruction State Checker Fails
        Returns:
        References:
        Examples:

    """

    def __init__(self):
        super(MPCNetwork, self).__init__()
        # encrypt layers with self.MPCConjugateLayer = MPC.encryptPublicConjugateLayer(Network.PublicConjugateLayer)

    def main(self):
        """Compute 3-Party MPC Training for MPCNetwork. Encrypt model layers/gradients, and train with respect to mpc protocol."""
        raise NotImplementedError

    def encrypt_all(self):
        raise NotImplementedError

    def decrypt_all(self):
        raise NotImplementedError
