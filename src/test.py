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

from utils import *
from client import *

warnings.filterwarnings("ignore")

# todo: test the corruptions for corruption regularization
# todo: setup exp configs; hardcode the graphs (x-y axis) that will be made based on the notes you have in dynalist and write the pseudocode in terms of matplotlib.pyplot if necessary

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
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = tf.cast(x_train, dtype=tf.float32)
    x_test = tf.cast(x_test, dtype=tf.float32)
    
    # process the same dataset
    return (x_train[idx * 5000 : (idx + 1) * 5000], y_train[idx * 5000 : (idx + 1) * 5000])

def load_test_partition(idx : int):
    assert idx in range(10)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = tf.cast(x_train, dtype=tf.float32)
    x_test = tf.cast(x_test, dtype=tf.float32)
    return (x_test[idx * 1000 : (idx + 1) * 1000], y_test[idx * 1000 : (idx + 1) * 1000])

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