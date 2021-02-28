import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import random
import pickle

from nn.network import Network
from nn.metrics import Metrics

class MPC():
    '''
    MPC stores core logic, whereas MPCTensor encrypts the model layers and tensors (specific to tensorflow state graph), and the MPCNetwork will use functions from MPCTensor and MPC in order to build_and_compile_encrypted_model() which utilizes both dp and mpc protocol logic.
    '''
    def __init__(self):
        self.secret = []
        # setup shares for each secret for each secret is-a f_1 in vector set f
        self.secret = []
        self.party_1 = [] # each party computes a subset of the required shares for each secret iterating over all secrets to complete forward prop, iterate over each batch, iterate over defined epochs
        self.party_2 = []

    @staticmethod
    def initialize_network_secrets(network):
        raise NotImplementedError

    @staticmethod
    def getMPCMetrics():
        """Metrics.mpc_metric but nests all metrics retrieved from Metrics class that were computed from in network. Use during MPC training."""
        raise NotImplementedError
    
    @staticmethod
    def initialize_network_shares(network_secrets):
        raise NotImplementedError

    @staticmethod
    def encryptNetworkLayers(network):
        raise NotImplementedError
    
    @staticmethod
    def encryptInputTensor(network):
        raise NotImplementedError

    @staticmethod
    def encryptConvLayers(network):
        raise NotImplementedError

    @staticmethod
    def encryptGradients(network_gradients):
        raise NotImplementedError

    @staticmethod
    def decryptGradients(network_gradients):
        """De-noise gradients."""
        raise NotImplementedError
    
    @staticmethod
    def encrypt_network():
        # assert type, act as parent function to encrypt all layers         
        pass

    def reconstruct_shares(self):
        """reconstruct secret shares computed by n parties"""
        raise NotImplementedError 


class MPCTensor:

    """
        Description: Encrypt Tensors/Layers for MPCNetwork 
        Args: None
        Returns: InputTensor for tf.keras.Model.layer
        Raises:
        References:
        Examples:    
    
    """

    raise NotImplementedError

    

    

 
        








