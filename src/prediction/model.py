import keras
import os

import numpy as np
import random
import time
from tqdm import tqdm
from collections import deque


from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

import torch
from torch import nn
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, classes=19):
        # recursive inheritance
        super(Model, self).__init__()

    def build_model(self):
        # call torch network constructor
        network = self.models.vgg16(pretrained=True)


if __name__ == '__main__':
    model = Model()
    model.summary()
