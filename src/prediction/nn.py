import keras
import os

import numpy as np
import random
import time

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

import torch
from torch import nn
import torchvision.models as models

from PIL import Image
from torchvision import transforms

import logging

# configure to DEBUG state for logging level, format log records (data, time, and message)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)

# options: FileHandler, StreamHandler, HTTPHandler, NullHandler, SMTPHandler, SocketHandler, MemoryHandler




class Prediction(models.VGG):
    def __init__(self, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        # recursive inheritance
        super(Prediction, self).__init__()


    def build_model(self):
        # call torch network constructor
        network = self.models.vgg16(pretrained=True)


if __name__ == '__main__':
    model = Prediction()
