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

from federated.crypto_utils import *
from federated.crypto_network import *

'''
Definition 1:

  1. Admissibility Constraint: x
∗ ∈ X˜;

  2. Distance Constraint: D(µ(x, x∗
  ), α), and

  3. Target Behavior Constraint: A(x, x∗
  , β).

The Admissibility Constraint (1) ensures that the adversarial input x∗ belongs to the space of admissible perturbed
inputs. 

The Distance Constraint (2) constrains x∗ to be no more distant from x than α. 

Finally, the Target Behavior Constraint (3) captures the target behavior of the adversary as a predicate A(x, x∗, β) which is true if the adversary changes the behavior of the ML model by at least β modifying x to x∗. If the three constraints hold, then we say that
the ML model has failed for input x. We note that this is a so called “local” robustness property for a specific input x, as
opposed to other notions of “global” robustness to changes to a population of inputs (see Dreossi et al. [2018b]; Seshia et al. [2018].


Typically, the problem of finding an adversarial example x ∗ for a model f at a given input x ∈ X as formulated above, can be formulated as an optimization problem in one of two ways: • Minimizing perturbation: find the closest x ∗ that alters f’s prediction. This can be encoded in constraint (2) as µ(x, x∗ ) ≤ α; • Maximizing the loss: find x ∗ which maximizes false classification. This can be encoded in the constraint (3) as L(f(x), f(x ∗ )) ≥ β.
  - is it more relevant to certify robustness of the model to a set of perturbed/adversarial inputs than to focus on whether adversarial examples exist?

'''