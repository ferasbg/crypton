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

def create_table(header: list, csv):
    pass

def create_plot(x_axis, y_axis, x_data: list, y_data: list):
    pass

def plot_client_model(model):
    keras.utilsplot_model(model, to_file='model.png')
