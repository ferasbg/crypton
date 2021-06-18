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

def compute_admissibility_constraint(x):
  '''
  The Admissibility Constraint (1) ensures that the adversarial input x∗ belongs to the space of admissible perturbed inputs.
  
  Eq: x∗ ∈ X˜

  '''

  return x

def compute_distance_constraint(adv_step_size, adv_grad_norm):
  '''
  The Distance Constraint (2) constrains x∗ to be no more distant from x than α. 

  Eq: D(µ(x, x∗), α)

  '''
  x = []
  return x

def compute_target_behavior_constraint(a, x, x_prime, beta):
  '''
  The Target Behavior Constraint (3) captures the target behavior of the adversary as a predicate A(x, x∗, β) which is true if the adversary changes the behavior of the ML model by at least β modifying x to x∗. If the three constraints hold, then we say that
  the ML model has failed for input x. We note that this is a so called “local” robustness property for a specific input x, as
  opposed to other notions of “global” robustness to changes to a population of inputs (see Dreossi et al. [2018b]; Seshia et al. [2018].

  Eq: A(x, x∗, β)

  '''
  x = []
  return x