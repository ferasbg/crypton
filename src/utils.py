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
from flwr.server.strategy import (
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
import imagedegrade
from imagedegrade import np as degrade
from imagecorruptions import corrupt


class HParams(object):
    '''
        adv_multiplier: The weight of adversarial loss in the training objective, relative to the labeled loss.
        adv_step_size: The magnitude of adversarial perturbation.
        adv_grad_norm: The norm to measure the magnitude of adversarial perturbation.

    '''

    def __init__(self, num_classes, adv_multiplier, adv_step_size, adv_grad_norm, adv_reg_state=True):
        # store model and its respective train/test dataset + metadata in parameters
        self.input_shape = [28, 28, 1]
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
        self.gaussian_state : bool = False
        # if (params.gaussian_state): model = tf.keras.models.Model.add(params.gaussian_layer, stack_index=1)
        self.gaussian_layer = keras.layers.GaussianNoise(stddev=0.2)
        self.clip_value_min = 0.0
        self.clip_value_max = 1.0
        self.adv_reg_state = adv_reg_state

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
    - FedAdagrad helps server model converge on heteregeneous data better; that's all

    - iteratively use the subset of corruptions that can have psuedorandom noise vectors applied e.g. severity
    - non-uniform, non-universal perturbations to the image; how does this fare as far as 1) min-max perturbation in adv. reg. and 2) against universal, norm-bounded perturbations?
    - each config has 1 specific corruption applied along with structured signals for adv. regularization
    - each config also has 1 specific federated strategy of course
    - either way test all permutations iteratively
    - first get base config working before extending to parse_args
    - also define the setup so that parse_args can process correctly with simulation.py

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

    Usage:
        # for batch in train_set_for_adv_model:
        #     adv_model.perturb_on_batch(batch)
        #     for element in batch:
        #         element = Data.apply_noise_image_degrade(element, noisa_sigma=0.05)
        #         element = Data.apply_blur_corruption(element, "gaussian_blur")



    '''
    corruption_tuple = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
                    "glass_blur", "motion_blur", "zoom_blur", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
                    "jpeg_compression", "speckle_noise", "gaussian_blur", "spatter",
                    "saturate"]

    # applying image corruptions is less of a priority than setting up adv. reg client and seeing errors there and ironing vs. dataset specific reg.
    # the goal here is categorized corruptions; rather than just ad-hoc using every corruption there is
    @staticmethod
    def apply_misc_corruptions(image : np.ndarray, corruption_name : str) -> np.ndarray:
        # apply_misc_corruptions (lighting, env conditions, edited/filtered data) --> spatter, saturate, fog, brightness, contrast
        misc_corruption_set = ["spatter", "saturate", "fog", "brightness", "contrast"]
        for corruption_str in misc_corruption_set:
            if (corruption_name == corruption_str):
                image = imagecorruptions.corrupt(image, corruption_name=corruption_str, severity=1)

        return image

    @staticmethod
    def apply_blur_corruption(image : np.ndarray, blur_corruption_name : str) -> np.ndarray:
        # iter over blur corruptions
        # support a subset that is relevant to imperceptible fidelity change from source np.ndarray matrix distribution
        blur_corruptions = ["motion_blur", "glass_blur", "zoom_blur", "gaussian_blur", "defocus_blur"]
        return image


    @staticmethod
    def apply_data_corruption(image : np.ndarray, corruption_name : str) -> np.ndarray:
        # apply_data_corruptions --> jpeg_compression
        data_corruption_set = ["jpeg_compression", "elastic_transform", "pixelate"]
        return image

    @staticmethod
    def apply_noise_corruption(image : np.ndarray, corruption_name : str) -> np.ndarray:
        # apply_noise_corruptions --> gaussian noise (omit and figure out nsl backend implementation with tf.GradientTape as tape), shot noise, impulse noise, etc
        # iteratively use the subset of corruptions that can have psuedorandom noise vectors applied e.g. severity
        noise_corruption_set = ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"]
        return image

    @staticmethod
    def apply_noise_image_degrade(image : np.ndarray, noise_sigma : float):
        # noise_sigma specifies gaussian_noise_stdev
        image = imagedegrade.np.noise(image, noise_sigma)
        return image

    @staticmethod
    def image_compression_distortion(image : np.ndarray, intensity_range=0.1):
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

    @staticmethod
    def perturb_base_model_dataset(dataset, parameters : HParams):
        pass

    @staticmethod
    def apply_corruptions_to_dataset(dataset, model, corruption_name : str):
        # if tf.keras.Model --> assume dataset is a set of tuples --> convert back to dict then apply with Data.apply_corruption(dataset : Dict[np.ndarray])  then convert back to tuples
        # if AdversarialRegularization --> assume dataset is a set of dictionaries --> iterate over dict when applying to each image of type 'np.ndarray'
        pass

    @staticmethod
    def load_train_partition(train_partitions, idx: int):
        # return a tuple given the target client (index to iterate over all clients)
        # (x_train, y_train) are already partitioned with Data.create_train_partitions(). This is untested, but validate that the return type of Tuple is consistent with the logic written in DatasetConfig and for storing the partition for the client in ExperimentConfig and accessed by AdvRegClientConfig or ClientConfig.
        # usage: train_partitions[0][0] --> x_train for client 0
        client_train_partition = train_partitions[idx]
        return client_train_partition

    @staticmethod
    def load_test_partition(test_partitions, idx : int):
        client_test_partition = test_partitions[idx]
        return client_test_partition

    @staticmethod
    def create_train_partitions(x_train, y_train, num_clients : int):
        '''
        Usage:
            - train_partitions = create_train_partitions(dataset, num_clients=args.num_clients)
            - current_train_dataset = train_partitions[current_client_idx]

        Notes:
            - preprocess the partitions before wrapping them into the MapDataset / BatchDataset objects

        '''
        train_partitions = []
        if (num_clients == 10):
            for i in range(num_clients):
                # partition x_train and y_train based on the num_clients
                partition = []
                # 0:5000; 50000:10000, 10000:15000, 15000:20000, ..., 45000:50000
                x_train = x_train[(i * (50000/num_clients)) : (i + 1) * (50000/num_clients)]
                y_train = y_train[(i * (50000/num_clients)) : (i + 1) * (50000/num_clients)]
                partition = (x_train, y_train)
                train_partitions.append(partition)

        return train_partitions

    @staticmethod
    def create_test_partitions(x_test, y_test, num_clients : int):
        test_partitions = []
        if (num_clients == 10):
            for i in range(num_clients):
                # partition x_train and y_train based on the num_clients
                partition = []
                # 0:5000; 50000:10000, 10000:15000, 15000:20000, ..., 45000:50000
                x_test = x_test[(i * (50000/num_clients)) : (i + 1) * (50000/num_clients)]
                y_test = y_test[(i * (50000/num_clients)) : (i + 1) * (50000/num_clients)]
                partition = (x_test, y_test)
                test_partitions.append(partition)

        return test_partitions

    @staticmethod
    def perturb_dataset_partition(partition, adv_model : nsl.keras.AdversarialRegularization, params : HParams):
        '''
        Server-side dataset perturbation.
        '''
        for batch in partition:
            adv_model.perturb_on_batch(batch)
            # clip_by_value --> depends on sample implementation

        return partition    

    @staticmethod
    def get_mnist_image(image : np.ndarray):
        return tf.keras.preprocessing.image.array_to_img(image)

    @staticmethod
    def make_training_data_iid(dataset, num_partitions):
        # IID: data is shuffled, then partitioned into 100 clients with 500 train and 100 test examples per client
        return []

    @staticmethod
    def make_training_data_non_iid(dataset, num_partitions):
        # Non-IID: first sort the data, divide it into 200 shards of size 300 and assign 100 clients 2 shards
        return []

    @staticmethod
    def load_partition(idx: int):
        """Load 1/10th of the training and test data to simulate a partition."""
        assert idx in range(10)
        # process the same dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = tf.cast(x_train, dtype=tf.float32)
        x_test = tf.cast(x_test, dtype=tf.float32)

        return (
            x_train[idx * 5000 : (idx + 1) * 5000],
            y_train[idx * 5000 : (idx + 1) * 5000],
        ), (
            x_test[idx * 1000 : (idx + 1) * 1000],
            y_test[idx * 1000 : (idx + 1) * 1000],
        )