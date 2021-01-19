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

#logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)



class Prediction(models.VGG):
    """
    reference: src.prediction.nn.Prediction
    args:
        models.VGG (type: Object): store the VGG-16 DCNN for semantic image segmentation
    params:
        - self.batch_size
        - self.num_classes
        - self.image_size = [224,224]
        - self.mask_size
        - self.stride
        - self.input_channels = 3
        - self.middle_channels = self.input_channels / 2
        - self.output_channels
        - self.kernel_size = 3
        - self.stride = []
        - self.padding = []
        - self.feature_map
        - self.time_step
        - self.num_training
        - self.num_validation
        - self.train_generator
        - self.validation_generator
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
    Prediction()
