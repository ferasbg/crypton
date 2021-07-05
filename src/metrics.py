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

class Plot(object):
    def plot_certified_accuracy(outfile: str, title: str, max_radius: float, radius_step: float = 0.01) -> None:

        # for line in lines:
            # plot line with sns
            # plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

        # radius is the norm value eg adv_step_size

        plt.ylim((0, 1))
        plt.xlim((0, max_radius))
        plt.tick_params(labelsize=14)
        plt.xlabel("radius", fontsize=16)
        plt.ylabel("certified accuracy", fontsize=16)
        # plt.legend([method.legend for method in lines], loc='upper right', fontsize=16)
        plt.savefig(outfile + ".pdf")
        plt.tight_layout()
        plt.title(title, fontsize=20)
        plt.tight_layout()
        plt.savefig(outfile + ".png", dpi=300)
        plt.close()

    @staticmethod
    def save_plot(sns_plot):
        # check for directory, then add file to target directory
        if (os.path.isdir('/figures')):
            # add file to path
            os.walk("/figures")
            file = sns_plot.savefig("exp_config.png")

        else:
            os.makedir("figures")
            os.walk("/figures")
            file = sns_plot.savefig("exp_config.png")

    @staticmethod
    def create_sns_plot(num_rounds : int, server_metrics : tf.keras.callbacks.History):
        # setup the scale given num_rounds is 10 or 100
        pass

    @staticmethod
    def create_table(csv, norm_type="l-inf", options=["l-inf", "l2"]):
        # every set of rounds maps to a hardcoded adv_step_size, so we can measure this round set in terms of the adv_step_size set we want to iterate over
        headers = ["Model", "Adversarial Regularization Technique", "Strategy", "Server Model ε-Robust Federated Accuracy", "Server Model Certified ε-Robust Federated Accuracy"]
        # define options per variable; adv reg shares pattern of adversarial augmentation, noise (non-uniform, uniform) perturbations/corruptions/degradation as regularization
        adv_reg_options = ["Neural Structured Learning", "Gaussian Regularization", "Data Corruption Regularization", "Noise Regularization", "Blur Regularization"]
        strategy_options = ["FedAvg", "FedAdagrad", "FaultTolerantFedAvg", "FedFSV1"]
        metrics = ["server_loss", "server_accuracy_under_attack", "server_certified_loss"]
        # norm type used to define the line graph in the plot rather than an x or y axis label
        variables = ["epochs", "communication_rounds", "client_learning_rate", "server_learning_rate", "adv_grad_norm", "adv_step_size"]
        nsl_variables = ["neighbor_loss"]
        # measure severity as an epsilon when labeling the line graph in plot
        baseline_adv_reg_variables = ["severity", "noise_sigma"]
        table = pd.DataFrame(columns=headers)
        
    @staticmethod
    def plot_client_model(model):
        file = keras.utils.plot_model(model, to_file='model.png')
        save_path = '/media'
        file_name = "model.png"
        os.path.join(save_path, file_name)

    @staticmethod
    def plot_img(image : np.ndarray):
        plt.figure(figsize=(32,32))
        # iteratively get perturbed images based on their norm type and norm values (l∞-p_ε; norm_type, adv_step_size)
        plt.imshow(image, cmap=plt.get_cmap('gray'))

    @staticmethod
    def update_dataframe(dataframe : pd.DataFrame, row):
        '''
            sns.relplot(
                data=fmri, x="timepoint", y="signal", col="region",
                hue="event", style="event", kind="line",
            )

        '''
        exp_config_data = pd.DataFrame(row=row)
        dataframe.append(exp_config_data)
    
    @staticmethod
    def create_line_plot():
        # Reference: https://seaborn.pydata.org/generated/seaborn.lineplot.html#seaborn.lineplot
        sns.set_theme(style="whitegrid")
        rs = np.random.RandomState(365)
        values = rs.randn(365, 4).cumsum(axis=0)
        dates = pd.date_range("1 1 2016", periods=365)
        data = pd.DataFrame(values, dates, columns=["Model", "Adversarial Regularization Technique", "Strategy", "Server Model ε-Robust Federated Accuracy", "Server Model Certified ε-Robust Federated Accuracy"])
        return sns.lineplot(data=data, palette="tab10", linewidth=2.5)
