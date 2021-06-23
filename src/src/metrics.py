#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig

import argparse
import json
import logging
import math
import os
import random
import sys
import time
import tensorflow as tf
from tensorflow import keras
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def create_table(header: list, csv):
    pass

def create_plot(x_axis, y_axis, x_data: list, y_data: list):
    pass

def plot_client_model(model):
    keras.utils.plot_model(model, to_file='model.png')

def plot_perturbed_image(image):
    plt.figure(figsize=(21, 21))
    # iteratively get perturbed images based on their norm type and norm values (l∞-p_ε; norm_type, adv_step_size)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(image), cmap='gray')
    plt.axis('off')

'''
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) 
plt.show()

'''