import argparse
import collections
import logging
import os
import pickle
import random
import sys
import time
import warnings

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import tqdm
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
from PIL import Image
from tensorflow import keras

warnings.filterwarnings('ignore')


def model_fn():
    # build layers of public neural network to pass into tff constructor as plaintext model, do in function since it's static 
    model = Sequential()
    # feature layers
    model.add(Conv2D(32, (3, 3), activation='relu',
                        kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='relu',
                        kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu',
                        kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu',
                        kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu',
                        kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu',
                        kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    # classification layers
    model.add(Dense(128, activation='relu',
                    kernel_initializer='he_uniform'))
    # 10 output classes possible
    model.add(Dense(10, activation='softmax'))
    
    # other method of constructing federated network object
    input_spec = collections.OrderedDict(
            x=tf.TensorSpec(shape=[None, None], dtype=tf.float32),
            y=tf.TensorSpec(shape=[None, 1], dtype=tf.int64)  
        )
    # tff wants new tff network created upon instantiation or invocation of method call
    crypto_network =  tff.learning.from_keras_model(model, input_spec=input_spec, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])
    return crypto_network




