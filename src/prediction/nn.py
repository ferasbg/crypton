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

# configure to DEBUG state for logging level, format log records (data, time, and message)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)



class Prediction(models.VGG):
    """
    reference: src.prediction.nn.Prediction
    args:
        models.VGG (type: Object): store the VGG-16 DCNN for semantic image segmentation
    methods: Prediction.build_model(), Prediction.preprocess(), Prediction.train(), Prediction.compile_model()
    """

    def __init__(self, model='vgg16'):
        super(Prediction, self).__init__()


    def train(self):
        """
        name: src.Prediction.train()
        args:
            - arg 1 (type: ): description
            - arg 2 (type: ): description
        usage: description

        """
        pass


    def build_model(self):
        """
        name: src.Prediction.build_model()
        args:
            - arg 1 (type: ): description
            - arg 2 (type: ): description
        usage: description

        """
        network = models.vgg16(pretrained=True)
        network.train()


    def compile_model(self):
        """
        name: src.Prediction.compile_model()
        args:
            - arg 1 (type: ): description
            - arg 2 (type: ): description
        usage: description

        """
        pass

    def evaluate(self, network):
        network.eval()


if __name__ == '__main__':
    model = Prediction()
