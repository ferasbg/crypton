import numbers
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms, datasets

import torch.nn.functional as F
import warnings

def inputToTensor(self, input_image):

        """
        Convert input frames into Tensor object. Apply transformations given variables of input image. Pass 1024x2048 images in terms of 224x224 dimensions. Crop, then take set of 224x224 matrices of input image, then compute on the Tensor that stores the image and its encoded and transformed pixelwise data.

        """

        pass




def load_model():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
    return model


def check_layers(model):
    """
    precedence property 1: verify network weight state (if member variables are initialized correctly) with getattr()

    """
    raise NotImplementedError

def check_weights():
    """
    precedence property 2: verify network state (if member variables are initialized correctly) with getattr()
    """

    raise NotImplementedError


def preprocess():
    """Pre-process data and pixel-categories. Load data and apply transformations to data, and save to path"""
    raise NotImplementedError

# setup directory
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def save_model(model):
    # save as model state dict
    pass
