import argparse
import flwr
from keras.metrics import sparse_categorical_accuracy
from neural_structured_learning.keras.adversarial_regularization import AdversarialRegularization
import tensorflow as tf
from tensorflow import keras
from keras import layers
import neural_structured_learning as nsl
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
from keras.regularizers import l2
import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler
from flwr.server.strategy import FedAdagrad, FedAvg, FaultTolerantFedAvg, FedFSv1, FastAndSlow
import bokeh
import seaborn as sns
import pandas as pd
import chartify
import matplotlib.pyplot as plt
import json
import jsonify
import logging
from logging import Logger
from keras.callbacks import History, EarlyStopping

# purpose: create plots given the unpacked plot data collected and saved as .pkl files
# initially label and print out the data to then combine them together for the final plots. First do this within a limited set of rounds. We will do this in terms of 1000 rounds.

'''
    # plot loss (regularized --> adversarial)

    plt.plot(history.history['loss'], label='training loss')
    plt.legend(['Training Loss'])
    plt.show()
    plt.savefig('client_train_loss')

    ## plot accuracy (sparse_categorical)

    plt.plot(history.history['accuracy'])
    plt.title('Client Model Regularization Accuracy')
    plt.legend(['Training Accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()
    plt.savefig('client_train_accuracy')

'''

plot_vars = ["Client Model Type", "Adversarial Regularization Technique", "Federated Strategy", "Server Model ε-Robust Federated Accuracy Under Attack"]
adv_reg_types = ["Neural Structured Learning", "Gaussian Regularization", "Data Corruption Regularization", "Noise Regularization", "Blur Regularization"]
# mathematical formulation must inter-weave and relate the federated strategy algorithm (equation set) to the neural structured learning algorithm relative to the federated system 
adaptive_strategy_options = ["FedAdagrad"]
non_adaptive_strategy_options = ["FedAvg", "FaultTolerantFedAvg", "FedFSV1"]
metrics = ["server_loss", "server_accuracy_under_attack", "server_certified_loss"]
dependent_variables = ["epochs", "communication_rounds", "client_learning_rate", "server_learning_rate", "adv_grad_norm", "adv_step_size"]
nsl_variables = ["adversarial_neighbor_loss"]
# the severity epsilon will be constant, as well as the norm type across adv. regularization techniques
baseline_adv_reg_variables = ["severity", "noise_sigma"]
# not a focus for paper 1, but rather paper 2
certification_methods = ["distance_constraint", "min_max_perturbation_test", "admissibility_constraint", "target_behavior_constraint"]