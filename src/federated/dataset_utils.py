import argparse
import collections
import logging
import os
import pickle
import random
import sys
import time
import warnings

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import tqdm
from keras import backend as K
from keras import optimizers, regularizers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.datasets import cifar10
from keras.datasets.cifar10 import load_data
from keras.layers import (Activation, BatchNormalization, Conv2D,
                          Conv2DTranspose, Dense, Dropout, Flatten,
                          GaussianDropout, GaussianNoise, Input, MaxPool2D,
                          ReLU, Softmax, UpSampling2D)
from keras.layers.core import Lambda
from keras.models import Input, Model, Sequential, load_model, save_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow import keras
# import image transformations library here, which works with cifar-100 specific images

def apply_random_transformations(state, image):
    # this is specific to the image data rather than how the image data is processed, so it will be used on the cifar-100 dataset in main.main
    # state implies requirement to apply transformations to the dataset, which we need to know how to access each individual image to pass during partition and processing
    if (state):
        return 0
    else:
        return 1

def apply_image_corruptions(image):
    pass

# there's a list in the "Flower Framework" paper that states all the conditions for the image data
# yes:  should we have subsets that factor in real-world data, perturbations, and gaussian noise? to see what client model converges the fastest? Should we then isolate copies of the client networks that fit the same training parameters to then compare the server model inferences?

def apply_all_transformations(image):
    image = apply_image_corruptions()
    image = apply_random_transformations()

def partition_dataset():
  pass

def setup_dataset():
    pass
