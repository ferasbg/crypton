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

class Crypto():
    '''
    Crypto stores logics for differential privacy and federated learning techniques. If secrets are computed as subsets of the entire composition function (network) f(x), then we must apply this to the context of a federated setting.

    '''

    @staticmethod
    def initialize_network_secrets(network):
        '''Setup secrets such that for f_n(x) = {f_1, ..., f_n}, each secret is a function over each layer given the shares e.g. weights and synapses of the previous layers.'''
        raise NotImplementedError

    @staticmethod
    def initialize_network_shares(network_secrets):
        raise NotImplementedError

    @staticmethod
    def encryptNetworkLayers(network):
        raise NotImplementedError

    @staticmethod
    def data_processor(dataset):
        raise NotImplementedError

    @staticmethod
    def encryptGradients(network_gradients):
        raise NotImplementedError

    @staticmethod
    def decryptGradients(network_gradients):
        """De-noise gradients."""
        raise NotImplementedError
    

    

    

 
        








