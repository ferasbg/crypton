import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import random
import pickle

from nn.network import Network

class MPCTensor:

    """
        Description: Encrypted InputTensor for MPCNetwork
        Args: None
        Returns: InputTensor for tf.keras.Model.layer
        Raises:
        References:
        Examples:    
    
    """

    def __init__(self):
        self.share0 = np.randn(.001)
        self.share1 = np.randn(.001)
        self.secret = {}
        self.party_1 = {}
        self.party_2 = {}

    def reconstruct(self):
        """reconstruct secret shares computed by n parties"""
        pass


    def encrypt(self):
        pass

    def decrypt(self):
        pass


class PublicTensor:
    """
        Description: PublicTensor=Default
        Args: None
        Returns: InputTensor for tf.keras.Model.layer
        Raises:
        References:
        Examples:    
    
    """
    def __init__(self):
        self.share0 = np.randn(.001)
        self.share1 = np.randn(.001)
        self.secret = {}
        self.party_1 = {}
        self.party_2 = {}
        

if __name__ == '__main__':
    MPCTensor()
    PublicTensor()










