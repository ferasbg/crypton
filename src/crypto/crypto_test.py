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

crypto_network = CryptoNetwork()
print(crypto_network.build_compile_crypto_model())
