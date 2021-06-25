
import argparse
import collections
import logging
import multiprocessing
import os
import pickle
import random
import sys
import threading
import time
import traceback
import warnings
from multiprocessing import Process
from typing import Dict, List, Tuple

import art
import cleverhans
import flwr as fl
import keras
import matplotlib.pyplot as plt
import neural_structured_learning as nsl
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
# FedAvg (Baseline); FedAdagrad (Comparable), FedOpt (Optimized FedAdagrad and Comparable)
from flwr.server.strategy import (FaultTolerantFedAvg, FedAdagrad, FedAvg,
                                  FedFSv1, Strategy, fedopt)
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
from tensorflow.keras import layers
from tensorflow.python.keras.backend import update
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.ops.gen_batch_ops import Batch
import inspect
import dataset
from adv_reg_simulation import HParams
import warnings
warnings.filterwarnings('ignore')

# data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x_test.shape) # (10000, 32, 32, 3) --> numpy arrays storing image data for there's 10000 images vectorized in the shape of 32x32 with 3 rgb dimensions
print(y_test.shape) # (10000, 1) --> numpy arrays containing indices to indicate labels

def read_train_data():
    for key, value in enumerate(x_train):
        # given eager tensor
            # each element in the stack or tuple in x_train is a tf.Tensor with a numpy array
        print(x_train[key]) 
