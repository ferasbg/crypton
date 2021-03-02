import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import random
import pickle
import tensorflow_federated as tff
from crypto_network import CryptoNetwork

'''
Crypto stores logics for differential privacy and federated learning techniques. If secrets are computed as subsets of the entire composition function (network) f(x), then we must apply this to the context of a federated setting.

'''

crypto_network = CryptoNetwork()


