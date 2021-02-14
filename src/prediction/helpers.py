import numbers
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

import warnings

class Dataset():
    raise NotImplementedError



def inputToTensor():
        # setup DataLoader for train dataset

        # load input image

        # for each image in the train dataset, apply transformations, and convert input_image to Tensor

        # return void


        """
        Convert input frames into Tensor object. Apply transformations given variables of input image. raise NotImplementedError 1024x2048 images in terms of 224x224 dimensions. Crop, then take set of 224x224 matrices of input image, then compute on the Tensor that stores the image and its encoded and transformed pixelwise data.

        """

        raise NotImplementedError






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
    raise NotImplementedError
