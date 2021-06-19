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
from neural_structured_learning import nsl
from PIL import Image
from tensorflow import keras
from tensorflow.python.keras.backend import update
from tensorflow.python.keras.engine.sequential import Sequential
import imagedegrade
from imagedegrade import np as degrade
from model import Network
from imagecorruptions import corrupt

class Data:
    '''
    This class handles processing data corruption, data preprocessing, and data utilities.

    - goal: using image processing techniques as a form of regularization through the data sent through the client models to evaluate better on real-world "adversarial examples". Structured signals from adv. regularization relation to entropy and adaptive federated optimization and effects from combined methods of data "regularization" to generate robust adversarial examples for the purpose of building a robust server-client model infrastructure.
    - process: perturb dtype=uint8 x_train before it's casted to tf.float32
    - note: modifying image fidelity as a means of regularization through the data and not the model
    - note: Perturbation Theory is relevant if we view the lense of our network through a dynamical systems perspective and evaluate it on those terms, also with respect to its adversarial robustness (its components contributions to it at the very least)
    - note: u can either generalize well to optimized resolution passed to ur model or fit well to the existing data that was damaged
    - note: if we can apply random transformations with a gaussian distribution ALONG with randomness, I think the model will do a lot better with a norm-bounded and mathematically fixed perturbation radius for each scalar in the image codec's matrix
    - note: use np.random.seed() to generate a seed value for noise vector to apply 
    - note: image degradation and corruptions ARE perturbations/transformations etc. (perhaps by the means of distortion, blur, etc)
    - question: how do ppl generally map the gradients of the network with its image data processed to maximize its loss? fgsm
    - question: how do ppl tell the difference between how model architecture affects robustness ? under an attack of course and assuming adv. reg. in training
    - common corruptions:
    - surface variations within corruptions for adv. reg. via data
    - goal: use corruptions/transformations/perturbations as adv.regularization other than nsl internal methods and GaussianNoise layer
    - note: relate image geometric transformations and map with structured signals with adv. reg. to relate to adaptive fed optimizer when aggregating updated client gradients (.... some middle steps though)

    References:
        @article{michaelis2019dragon,
        title={Benchmarking Robustness in Object Detection: 
            Autonomous Driving when Winter is Coming},
        author={Michaelis, Claudio and Mitzkus, Benjamin and 
            Geirhos, Robert and Rusak, Evgenia and 
            Bringmann, Oliver and Ecker, Alexander S. and 
            Bethge, Matthias and Brendel, Wieland},
        journal={arXiv preprint arXiv:1907.07484},
        year={2019}
        }

    In leu of structured signals, graph representations, and graph learning, here's a reference for the paper nsl depends on:

    "Parameterization invariant regularization, on the other hand, does not suffer from such a problem. In more precise terms, by parametrization invariant regularization we mean the regularization based on an objective function L(θ) with the property that the corresponding optimal distribution p(X; θ ∗ ) is invariant under the oneto-one transformation ω = T(θ), θ = T −1 (ω). That is, p(X; θ ∗ ) = p(X; ω ∗ ) where ω ∗ = arg minω L(T −1 (ω); D). VAT is a parameterization invariant regularization, because it directly regularizes the output distribution by its local sensitivity of the output with respect to input, which is, by definition, independent from the way to parametrize the model."

    '''

    @staticmethod
    def apply_imperceptible_pseudorandom_image_corruption(image: np.ndarray, corruption_name):
        corruption_tuple = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
                    "glass_blur", "motion_blur", "zoom_blur", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
                    "jpeg_compression", "speckle_noise", "gaussian_blur", "spatter",
                    "saturate"]
        # iteratively use the subset of corruptions that can have psuedorandom noise vectors applied e.g. severity
        # if corruption_name explicitly defined and specified from parse_args() (later, rn it'll be explicitly defined), then apply that specific corruption iteratively on all the data (iteration defined in simulation.py)
        # 1 corruption per exp config
            # for image in x_train: apply_imperceptible_pseudorandom_image_corruption(image, corruption_name)
        # non-uniform, non-universal perturbations to the image; how does this fare as far as 1) min-max perturbation in adv. reg. and 2) against universal, norm-bounded perturbations?
        return image

    @staticmethod
    def apply_noise(image : np.ndarray, noise_sigma : float):
        # noise_sigma specifies gaussian_noise_stdev
        image = imagedegrade.np.noise(image, noise_sigma)
        return image

    @staticmethod
    def image_compression_distortion(image : np.ndarray, intensity_range=0.1):
        # distortion is not the same as data/resolution loss
        jpeg_quality = 85 # 85% distortion
        image = imagedegrade.np.jpeg(input=image, jpeg_quality=jpeg_quality, intensity_range=intensity_range)
        return image

    @staticmethod
    def cast_data_to_float32(x_set):
        '''
            Usage:
                x_train, x_test = cast_data_to_float32(x_train), cast_data_to_float32(x_test)

        '''
        x_set = tf.cast(x_set, dtype=tf.float32)
        return x_set

