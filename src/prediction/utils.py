# statistical metrics for segnet

import os
import re
import torch
from torch import torch.nn as nn
from torch.autograd import Variable
import cv2
import numpy as np
import matplotlib.pyplot as plt


def batch_norm():
    pass

## data preprocessing

def normalize():
    pass

def denormalize():
    pass

## stats
def compute_loss():
    pass

def compute_softmax():
    pass

def compute_accuracy():
    """ Compute Cross-Entropy Loss Given compute_softmax(Model model) """
    pass

## recursive function to evaluate FCN
def eval(model):
    model.eval()


def save_model(model):
    """ save trained model weight & architecture binaries """
    PATH = "./model.pt"
    torch.save(model, PATH)

def load_model(model, PATH):
    model = torch.load(PATH)