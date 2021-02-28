import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import random
import pickle
import tf_encrypted as tfe

from network import Network

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

    def main(self):
        """Compute 3-Party MPC Training for MPCNetwork. Encrypt model layers/gradients, and train with respect to mpc protocol."""
        raise NotImplementedError

    def reconstruct(self, share0, share1):
        """Reconstruct computations, probably iterate as a subset set iterating over the kernels of the input matrix."""
        return (share0 + share1) % .0001

    def encrypt(self):
        raise NotImplementedError

    def decrypt(self):
        raise NotImplementedError

    def train_mpc(self):
        raise NotImplementedError
