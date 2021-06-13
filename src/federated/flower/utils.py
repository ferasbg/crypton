#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig
import argparse
import collections
import logging
import os
import pickle
import random
import sys
import time
import warnings
from typing import List, NamedTuple, Tuple

import flwr
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_federated as tff
import tensorflow_privacy as tpp
import tensorflow_probability as tpb
import tqdm
from federated.crypto_network import Client
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes
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
from model import Network
from PIL import Image
from tensorflow import keras
from tensorflow.python.keras.engine.sequential import Sequential

def apply_random_image_transformations(image : np.ndarray):
    # this is specific to the image data rather than how the image data is processed, so it will be used on the cifar-100 dataset in main.main
    pass

def apply_image_corruptions(image : np.ndarray):
    return image

def fault_tolerant_federated_averaging():
    pass
