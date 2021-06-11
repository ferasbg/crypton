import os
import random
import sys
import numpy
import sympy
import scipy

import tensorflow as tf
import keras
from federated.crypto_network import CryptoNetwork, Client, TrustedAggregator
from model import Network 
import tensorflow_federated as tff 
import tensorflow_privacy as tpp 

def main():
    # two types of client models, one with defenses to optimize network to fit to adversarial attacks, the other is isolated to perturbation attacks only
    defensive_client_model = Client(defense_state=True)
    base_client_model = Client(defense_state=False)

if  __name__ == '__main__':
    # run this process with flwr on sagemaker; test with very low config numbers to get things working first
    main()