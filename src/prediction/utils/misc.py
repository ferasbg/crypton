"""
name: src.prediction.utils
description: store helper methods and setup configuration and initialization for core compute of neural net class
"""
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def preprocess_frames(self):
    """
        description: pre-process input data for network
        args (@param):
        returns (@return): (type: Object)
    """
    pass


def batch_norm():
    pass

def normalize():
    pass

def denormalize():
    pass

# setup directory
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)




def save_model(model):
    """ save trained model weight & architecture binaries """
    PATH = "./model.pt"
    torch.save(model, PATH)

def load_model(model, PATH):
    model = torch.load(PATH)

