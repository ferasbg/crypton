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
from PIL import Image
from tensorflow import keras
from tensorflow.python.keras.backend import update
from tensorflow.python.keras.engine.sequential import Sequential

from model import Network

class Perturbation():
    '''
        Usage: 
            image_set : partition of image data per client # check if there's anything diff to do btwn. cifar-100 and cifar-10
            perturbation_dataset = Perturbation(image_set)            

        Discussion:
            - If we a train a network to fit to noise-like perturbations based on the gaussian-distribution, is our adversarial example "robust" such that our network can evaluate with greater robustness when given other perturbation types differentiated by norms. 
            - Should we have a changing perturbation applied to the elements in our dataset even if we know that our model will converge either way?
            - What conditions of the perturbation with respect to lp norm and its type (affect the process of the network)
            - Could perturbations create "adversarial" overfitting, so how is this avoided?
            - Things get fun here where we scope down our perturbations to types we can evaluate other than their norm (distance metric: l-inf, l-2) and epsilon value 
            - Perturbation Theory is relevant if we view the lense of our network through a dynamical systems perspective and evaluate it on those terms, also with respect to its adversarial robustness (its components contributions to it at the very least)
    '''
    def __init__(self, image_set : Tuple, norm_type : str):
        self.dataset = image_set
        self.norm_type = norm_type
        self.l2_eps = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.l_inf_eps = [
        0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
        # compute all the perturbation types at the same time upon instantiation
        self.norm_bounded_perturbation_dataset = [] # self.create_norm_bounded_perturbation_dataset()
        self.hamiltonian_perturbation_dataset = []
        self.brightness_perturbation_dataset = []
        self.mixed_perturbation_dataset = [] # different perturbation types

    @staticmethod
    def brightness_perturbation_norm(input_image):
        # applying image perturbations isn't in a layer, but rather before data is processed into the InputSpec and Input layer of the tff model
        sigma = 0.085
        brightness_threshold = 1 - sigma
        input_image = tf.math.scalar_mul(brightness_threshold, input_image)
        return input_image

    @staticmethod
    def apply_random_image_transformations(image : np.ndarray):
        # this is specific to the image data rather than how the image data is processed, so it will be used on the cifar-100 dataset in main.main
        pass

    @staticmethod
    def apply_image_corruptions(image : np.ndarray):
        return image

    @staticmethod
    def apply_image_degradation(image : np.ndarray):
        # image degradation WITH gaussian distribution acts as approach to dynamically adapt to chaos/randomness relating to realistic scenarios with image data
        # u can either generalize well to optimized resolution passed to ur model or fit well to the existing data that was damaged
        # apply image corruption, perturbation, compression loss; if we can apply random transformations with a gaussian distribution ALONG with randomness, I think the model will do a lot better with a norm-bounded and mathematically fixed perturbation radius for each scalar in the image codec's matrix
        image = Perturbation.apply_image_corruptions(image)
        image = Perturbation.apply_random_image_transformations(image)
        image = Perturbation.apply_resolution_loss(image)
        return image

    @staticmethod
    def apply_resolution_loss(image : np.ndarray):
        return image

