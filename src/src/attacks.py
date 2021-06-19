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
from typing import Dict, List, Tuple

import art
import cleverhans
import flwr as fl
import keras
import matplotlib.pyplot as plt
import numpy
import numpy as np
import scipy
import sympy
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_federated as tff
import tensorflow_privacy as tpp
import tensorflow_probability as tpb
import tqdm
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import (  # FedProx; FedAdagrad helps convergence behavior which in turn helps optimize model robustness; fedOpt is configurable Adagrad for server-side optimizations for the server model e.g. trusted aggregator
    FaultTolerantFedAvg, FedAdagrad, FedAvg, FedFSv1, Strategy, fedopt)
from keras import backend as K
from keras import optimizers, regularizers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.datasets import cifar10, cifar100
from keras.datasets.cifar10 import load_data
from keras.layers import (Activation, BatchNormalization, Conv2D,
                          Conv2DTranspose, Dense, Dropout, Flatten,
                          GaussianDropout, GaussianNoise, Input, MaxPool2D,
                          ReLU, Softmax, UpSampling2D)
from keras.layers.core import Lambda
from keras.models import Input, Model, Sequential, load_model, save_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import neural_structured_learning as nsl
from PIL import Image
from tensorflow import keras
from tensorflow.python.keras.backend import update
from tensorflow.python.keras.engine.sequential import Sequential

import cv2
from model import Network

'''

Generate adversarial examples.

Methods will include:
    -  imperceptible ASR attack
    - universal perturbation attack (norm)
    - deepfool
    - PixelAttack
    - target universal perturbation attack
    - projected gradient descent attack
    - fast gradient sign method
    - hamiltonian pixelwise perturbations

'''

def setup_sample_data_for_file():
    # setup dataset and store input image to test
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    # take first image from train set
    image = x_train[-1:]
    return image, image.shape, tf.keras.preprocessing.image.array_to_img(image)

def brightness_perturbation_norm(input_image):
    sigma = 0.085
    brightness_threshold = 1 - sigma
    input_image = tf.math.scalar_mul(brightness_threshold, input_image)
    return input_image

def hamiltonian_perturbation(input_image):
    return input_image

def additive_perturbation(input_image, norm_type, norm_value):
    return input_image

def apply_eval_perturbation(x_test):
    return x_test


if __name__ == '__main__':
    # setup data for testing each function
    setup_sample_data_for_file()
    # apply specified perturbation to use during evaluation (inference time, so to speak)
