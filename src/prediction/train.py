import logging
import os
import random
import time

import keras
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from torch import nn
from torchvision import transforms


from network import Network
import helpers


class Train(Network):
    """
    Verifiably Robust Training And Dev(Train)/Validation(Test) for Decrypted Network. Sub-class instance of `nn.Module` and `prediction.network.Network`
    Args:
        - 
        -
        -
        -

    Raises:
        -
        -


    Returns:
        -
        -
        -

    References:
        -
        -
        -



    """

    def __init__(self):
        super(Network, self).__init__()
        self.model = load_model()
        self.epochs = 1000
        self.batch_size = 64 # maybe 128
        self.learning_rate = .003



    def train(self):
        network = models.vgg16(pretrained=True)
        network.train()


if __name__ = '__main__':
    Train()
    Train.train()



