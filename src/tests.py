import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
import argparse
import flwr
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import (FaultTolerantFedAvg, FedAdagrad, FedAvg,
                                  FedFSv1, Strategy, fedopt)
import keras
import matplotlib.pyplot as plt
import neural_structured_learning as nsl
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers, optimizers, regularizers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.datasets import cifar10, cifar100
from keras.datasets.cifar10 import load_data
from keras.layers.core import Lambda
from keras.models import Input, Model, Sequential, load_model, save_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.backend import update
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.ops.gen_batch_ops import Batch
from keras import backend as K
import pytest
import cv2
from imagecorruptions import corrupt

from utils import *
from client import *

warnings.filterwarnings("ignore")

def build_base_server_model(num_classes : int):
    input_layer = layers.Input(shape=(28, 28, 1), batch_size=None, name="image")
    conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
    batch_norm = layers.BatchNormalization()(conv1)
    dropout = layers.Dropout(0.3)(batch_norm)
    conv2 = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform',  padding='same')(dropout)
    maxpool1 = layers.MaxPool2D((2,2))(conv2)
    conv3 = layers.Conv2D(64, (3,3), activation='relu',  kernel_initializer='he_uniform', padding='same')(maxpool1)
    conv4 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv3)
    maxpool2 = layers.MaxPool2D((2,2))(conv4)
    conv5 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(maxpool2)
    conv6 = layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv5)
    maxpool3 = layers.MaxPool2D((2,2))(conv6)
    flatten = layers.Flatten()(maxpool3)
    # flatten is creating error because type : NoneType
    dense1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flatten)
    # possibly remove defined kernel/bias initializer, but functional API will check for this and removes error before processing model architecture and config
    output_layer = layers.Dense(num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')(dense1)
    model = keras.Model(inputs=input_layer, outputs=output_layer, name='client_model')
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model

def load_train_partition(idx: int):
    # the declaration is in terms of a tuple to the assignment with the respective load partition function
    assert idx in range(10)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = tf.cast(x_train, dtype=tf.float32)
    x_test = tf.cast(x_test, dtype=tf.float32)

    # process the same dataset
    return (x_train[idx * 5000 : (idx + 1) * 5000], y_train[idx * 5000 : (idx + 1) * 5000])

def load_test_partition(idx : int):
    assert idx in range(10)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = tf.cast(x_train, dtype=tf.float32)
    x_test = tf.cast(x_test, dtype=tf.float32)
    return (x_test[idx * 1000 : (idx + 1) * 1000], y_test[idx * 1000 : (idx + 1) * 1000])

def test_partition_functions():
    for i in range(10):
        print("iterating over the train and test partition creation....")
        x_train, y_train = load_train_partition(idx=i)
        assert len(x_train) == 5000
        if (len(x_train) == 5000 and len(y_train) == 5000):
            print("the train data has been partitioned")
        assert len(y_train) == 5000
        x_test, y_test = load_test_partition(i)
        assert len(x_test) == 1000
        assert len(y_test) == 1000

        if (len(x_test) == 1000 and len(y_test) == 1000):
            print("the test data has been partitioned")


        train_data = tf.data.Dataset.from_tensor_slices({'image': x_train, 'label': y_train}).batch(32)
        val_data = tf.data.Dataset.from_tensor_slices({'image': x_test, 'label': y_test}).batch(32)

    xx_train, yy_train = load_train_partition(idx=0)
    print(len(xx_train), len(yy_train))

def load_partition(idx : int):
        assert idx in range(10)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        return (
            x_train[idx * 5000 : (idx + 1) * 5000],
            y_train[idx * 5000 : (idx + 1) * 5000],
        ), (
            x_test[idx * 1000 : (idx + 1) * 1000],
            y_test[idx * 1000 : (idx + 1) * 1000],
        )

datasets = tfds.load('mnist')
map_train_dataset = datasets['train']
map_test_dataset = datasets['test']
train_dataset_for_base_model = map_train_dataset.map(normalize).shuffle(10000).batch(32).map(convert_to_tuples)
test_dataset_for_base_model = map_test_dataset.map(normalize).batch(32).map(convert_to_tuples)

# partition > partition[0] > partition[0][i] for i in range(len(partition[0])) in DatasetConfig (partitions.append(train_partition and test_partition) and for partition in partitions: train = resize(train), test = resize(test)

# misc: smoothing, min-max perturbation, loss maximization as contradiction
corruptions = ["shot_noise", "impulse_noise", "defocus_blur",
                "glass_blur", "motion_blur", "zoom_blur", "elastic_transform", "pixelate",
                "jpeg_compression", "gaussian_blur"]


## either figure out how to pad 1d tensors or re-write the entire client dataset process with the model and processing edit

# when specifying corruptions in DatasetConfig
def corrupt_train_partition(train_samples, corruption_name: str):
    for i in range(len(train_samples)):
        train_samples[i] = imagecorruptions.corrupt(train_samples[i], corruption_name=corruption_name)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
train_partition = load_train_partition(0)
blur_corruption_set = ["motion_blur", "glass_blur", "zoom_blur", "gaussian_blur", "defocus_blur"]
data_corruption_set = ["jpeg_compression", "elastic_transform", "pixelate"]
noise_corruption_set = ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"]

# for i in range(len(x_train)):
#     for corruption_name in data_corruption_set:
#         print("corrupting data....")
#         x_train[i] = imagecorruptions.corrupt(x_train[i], corruption_name=corruption_name)

#     for corruption_name in blur_corruption_set:
#         print("corrupting blur....")

#         x_train[i] = imagecorruptions.corrupt(x_train[i], corruption_name=corruption_name)

#     for corruption_name in noise_corruption_set:
#         print("corrupting noise....")
#         x_train[i] = imagecorruptions.corrupt(x_train[i], corruption_name=corruption_name)

# # this works
# for i in range(len(x_train)):
#     for corruption in imagecorruptions.get_corruption_names():
#         print("corrupting image in train_sample set using" + "" + corruption)
#         x_train[i] = corrupt(x_train[i], corruption_name=corruption)

# setup dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
train_data = tf.data.Dataset.from_tensor_slices({'image': x_train, 'label': y_train}).batch(32)
val_data = tf.data.Dataset.from_tensor_slices({'image': x_test, 'label': y_test}).batch(32)

# setup parser
client_parser = setup_client_parse_args()
args = client_parser.parse_args()
# the data is partitioned such that 
dataset_config = DatasetConfig(args=args)
# why are the types empty? is the model not fit to the data?
print(len(dataset_config.train_data))
print(type(dataset_config.train_data))
print(len(dataset_config.val_data))
print(type(dataset_config.val_data))
