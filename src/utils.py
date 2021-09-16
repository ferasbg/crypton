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
import imagecorruptions
import imagedegrade
import keras
import matplotlib.pyplot as plt
import neural_structured_learning as nsl
import numpy
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sympy
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_federated as tff
import tensorflow_privacy as tpp
import tensorflow_probability as tpb
import tqdm
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import (FaultTolerantFedAvg, FedAdagrad, FedAvg,
                                  FedFSv1, Strategy, fedopt)
from imagecorruptions import corrupt
from imagedegrade import np as degrade
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

def plot_client_model(model):
    file = keras.utils.plot_model(model, to_file='model.png')
    save_path = '/media'
    file_name = "model.png"
    os.path.join(save_path, file_name)

def plot_img(image : np.ndarray):
    plt.figure(figsize=(32,32))
    # iteratively get perturbed images based on their norm type and norm values (l∞-p_ε; norm_type, adv_step_size)
    plt.imshow(image, cmap=plt.get_cmap('gray'))

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
        
class HParams(object):
    '''
    Args:
        adv_multiplier: The weight of adversarial loss in the training objective, relative to the labeled loss.
        adv_step_size: The magnitude of adversarial perturbation.
        adv_grad_norm: The norm to measure the magnitude of adversarial perturbation.

    Notes:
        - Note that a convolutional neural network is generally defined by a function F(x, θ) = Y which takes an input (x) and returns a probability vector (Y = [y1, · · · , ym] s.t. P i yi = 1) representing the probability of the input belonging to each of the m classes. The input is assigned to the class with maximum probability (Rajabi et. al, 2021).

    References:
            - https://arxiv.org/abs/1409.1556
    '''

    def __init__(self, num_classes, adv_multiplier, adv_step_size, adv_grad_norm):
        # store model and its respective train/test dataset + metadata in parameters
        # by default it's 28x28x1 but if specified, it can change to 32x32x3
        self.num_classes = num_classes
        self.conv_filters = [32, 32, 64, 64, 128, 128, 256]
        self.kernel_size = (3, 3)
        self.pool_size = (2, 2)
        self.num_fc_units = [64]
        self.batch_size = 32
        self.epochs = 25
        self.adv_multiplier = adv_multiplier
        self.adv_step_size = adv_step_size
        self.adv_grad_norm = adv_grad_norm  # "l2" or "infinity" = l2_clip_norm if "l2"
        self.gaussian_layer = keras.layers.GaussianNoise(stddev=0.2)
        self.clip_value_min = 0.0
        self.clip_value_max = 1.0
        self.callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        self.stride = 2
        self.padding = 1
        self.dilation = 1
        self.epochs = 500
        self.input_channels = 3
        self.output_channels = 64  # number of channels produced by convolution
        self.bias = False
        # stabilize convergence to local minima for gradient descent
        self.weight_decay_regularization = 0.003 
        self.momentum = 0.05  # gradient descent convergence optimizer

class Data:
    '''
    This class handles processing data corruption, data preprocessing, and data utilities.

    Notes:
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
        - goal: use corruptions/transformations/perturbations as adv.regularization other than nsl internal methods and GaussianNoise layer
        - note: relate image geometric transformations and map with structured signals with adv. reg. to relate to adaptive fed optimizer when aggregating updated client gradients (.... some middle steps though)
        - FedAdagrad helps server model converge on heteregeneous data better; that's all
        - non-uniform, non-universal perturbations to the image; how does this fare as far as 1) min-max perturbation in adv. reg. and 2) against universal, norm-bounded perturbations?
        - train and eval loss data matters when relating the ideas of the regularizer and strategy in terms of architecture choice

    Research Notes:
        - In leu of structured signals, graph representations, and graph learning, here's a reference for the paper nsl depends on:
        - "Parameterization invariant regularization, on the other hand, does not suffer from such a problem. In more precise terms, by parametrization invariant regularization we mean the regularization based on an objective function L(θ) with the property that the corresponding optimal distribution p(X; θ ∗ ) is invariant under the oneto-one transformation ω = T(θ), θ = T −1 (ω). That is, p(X; θ ∗ ) = p(X; ω ∗ ) where ω ∗ = arg minω L(T −1 (ω); D). VAT is a parameterization invariant regularization, because it directly regularizes the output distribution by its local sensitivity of the output with respect to input, which is, by definition, independent from the way to parametrize the model."

    References:
            
        Corruptions Reference: Hendrycks, Dan and Dietterich, Thomas G.
        Benchmarking Neural Network Robustness to Common Corruptions and
        Surface Variations
    
    '''

    corruption_tuple = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
                    "glass_blur", "motion_blur", "zoom_blur", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
                    "jpeg_compression", "speckle_noise", "gaussian_blur", "spatter",
                    "saturate"]


    @staticmethod
    def apply_blur_corruption(image, corruption_name : str):
        # iter over blur corruptions
        # photons, fidelity reference for image data and non-convex transformations
        # support a subset that is relevant to imperceptible fidelity change from source np.ndarray matrix distribution
        blur_corruptions = ["motion_blur", "glass_blur", "zoom_blur", "gaussian_blur", "defocus_blur"]
        image = corrupt(image, corruption_name=corruption_name, severity=1)
        return image

    @staticmethod
    def apply_data_corruption(image, corruption_name : str):
        # apply_data_corruptions --> jpeg_compression
        data_corruption_set = ["jpeg_compression", "elastic_transform", "pixelate"]
        image = corrupt(image, corruption_name=corruption_name, severity=1)
        return image

    @staticmethod
    def apply_noise_corruption(image, corruption_name : str):
        # args.gaussian_reg --> invokes gaussian_base_model with tf.keras.GaussianNoise(stddev=0.2) which is an equivalent of imagecorruptions.corrupt(image, corruption_name="gaussian_noise")
        # iteratively use the subset of corruptions that can have psuedorandom noise vectors applied e.g. severity
        noise_corruption_set = ["shot_noise", "impulse_noise", "speckle_noise"]
        image = corrupt(image, corruption_name=corruption_name, severity=1)
        return image

    @staticmethod
    def apply_noise_image_degrade(image, noise_sigma : float):
        # noise_sigma specifies gaussian_noise_stdev
        image = imagedegrade.np.noise(image, noise_sigma)
        return image

    @staticmethod
    def image_compression_distortion(image, intensity_range=0.1):
        # either process per batch or the entire dataset before it's processed into network's input layer
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

    @staticmethod
    def perturb_adv_model_dataset(model, dataset, parameters : HParams):
        IMAGE_INPUT_NAME = 'image'
        LABEL_INPUT_NAME = 'label'
        for batch in dataset:
            perturbed_batch = model.perturb_on_batch(batch)
            perturbed_batch[IMAGE_INPUT_NAME] = tf.clip_by_value(perturbed_batch[IMAGE_INPUT_NAME], 0.0, 1.0)

        return dataset
    
    def load_partition(self, idx : int):
        assert idx in range(10)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = tf.cast(x_train, dtype=tf.float32)
        x_test = tf.cast(x_test, dtype=tf.float32)

        return (
            x_train[idx * 5000 : (idx + 1) * 5000],
            y_train[idx * 5000 : (idx + 1) * 5000],
        ), (
            x_test[idx * 1000 : (idx + 1) * 1000],
            y_test[idx * 1000 : (idx + 1) * 1000],
        )

    # Data.load_train_partition_for_100_clients(idx=args.client_partition_idx) if args.num_clients = 100
    # Data.load_test_partition_for_100_clients(idx=args.client_partition_idx) if args.num_clients = 100
    @staticmethod    
    def load_train_partition_for_100_clients(idx: int):
        assert idx in range(100)
        # 500/100 train/test split per partition e.g. per client
        # create partition with train/test data per client; note that 600 images per client for 100 clients is convention; 300 images for 200 shards for 2 shards per client is another method and not general convention, but a test
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = tf.cast(x_train, dtype=tf.float32)
        x_test = tf.cast(x_test, dtype=tf.float32)
        # 5000/50000 --> 500/50000
        return (x_train[idx * 500: (idx + 1) * 500], y_train[idx * 500: (idx + 1) * 500])

    @staticmethod    
    def load_test_partition_for_100_clients(idx : int):
        assert idx in range(100) 
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = tf.cast(x_train, dtype=tf.float32)
        x_test = tf.cast(x_test, dtype=tf.float32)
        return (x_test[idx * 100: (idx + 1) * 100], y_test[idx * 100: (idx + 1) * 100])

    # MapDataset
    def normalize(features):
        features['image'] = tf.cast(
            features['image'], dtype=tf.float32) / 255.0
        return features

    def convert_to_tuples(features):
        return features['image'], features['label']

    def convert_to_dictionaries(image, label, IMAGE_INPUT_NAME='image', LABEL_INPUT_NAME='label'):
        return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}

    @staticmethod
    def make_training_data_non_iid(dataset, num_partitions):
        # Non-IID: first sort the data, divide it into 200 shards of size 300 and assign 100 clients 2 shards
        return []

class ExperimentConfig:
    @staticmethod
    def create_args_parser(client_partition_idx : int, adv_grad_norm : str, adv_multiplier=0.2, adv_step_size=0.05, batch_size=32, epochs=25, steps_per_epoch=None, num_clients=10, num_classes=10, model=None, nsl_reg : bool = False, gaussian_reg : bool = False, nominal_reg=True, corruption_name : str = "", client : str = "client", num_rounds=10, strategy="fedavg", fraction_fit=0.5, fraction_eval=0.2, min_fit_clients=2, min_eval_clients=10, dataset_config=None):
        # hardcode defaults given parameters.
        parser = argparse.ArgumentParser(description="Crypton Exp Config Object.")
        parser.add_argument("--client_partition_idx", type=int, required=False, default=client_partition_idx)
        parser.add_argument("--adv_grad_norm", type=str, required=False, default=adv_grad_norm)
        parser.add_argument("--adv_multiplier", type=float, required=False, default=adv_multiplier)
        parser.add_argument("--adv_step_size", type=float, required=False, default=adv_step_size)
        parser.add_argument("--batch_size", type=int, required=False, default=batch_size)
        parser.add_argument("--epochs", type=int, required=False, default=epochs)
        parser.add_argument("--steps_per_epoch", type=int, required=False, default=steps_per_epoch)
        parser.add_argument("--num_clients", type=int, required=False, default=num_clients)
        parser.add_argument("--num_classes", type=int, required=False, default=num_classes)
        parser.add_argument("--model", type=str, required=False, default=model)
        parser.add_argument("--nsl_reg", type=bool, required=False, default=nsl_reg)
        parser.add_argument("--gaussian_reg", type=bool, required=False, default=gaussian_reg)
        parser.add_argument("--nominal_reg", type=str, required=False, default=nominal_reg)
        parser.add_argument("--corruption_name", type=str, required=False, default=corruption_name)
        # options: "nsl_client", "client"
        parser.add_argument("--client", type=str, required=False, default=client)
        parser.add_argument("--num_rounds", type=int, required=False, default=3)
        parser.add_argument("--strategy", type=str, required=False, default="fedavg")
        parser.add_argument("--fraction_fit", type=float,
                            required=False, default=0.05)
        parser.add_argument("--fraction_eval", type=float,
                            required=False, default=0.5)
        parser.add_argument("--min_fit_clients", type=int,
                            required=False, default=10)
        parser.add_argument("--min_eval_clients", type=int,
                            required=False, default=2)
        parser.add_argument("--min_available_clients",
                            type=int, required=False, default=2)
        parser = parser.parse_args()
