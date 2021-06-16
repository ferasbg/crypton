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

import h5py
import torch


class FederatedMetrics:
    def __init__(self):
        '''
        - Update the state of variables for particular computations that can be graphed and stored into a table of use for the whitepaper.
        - parse_args, config, dev/run.sh, code cleanup/repo cleanup

        '''
        # get metrics (nominal --> formal; write setters)
        # get analytics per round per client set and iteration per configuration
        self.federated_accuracy_under_attack = 0
        self.natural_accuracy = 0
        self.federated_natural_accuracy = 0
        self.NUM_CLIENTS = 100
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 10
        # how does change in client learning rate to 0.02 affect server performance (possibly)
        self.CLIENT_LEARNING_RATE = 0.1
        self.SERVER_LEARNING_RATE = 0.1
        self.NUM_ROUNDS = 10
        self.CLIENTS_PER_ROUND = 10
        self.NUM_EXAMPLES_PER_CLIENT = 500
        self.CIFAR_SHAPE = (32, 32, 3)
        self.TOTAL_FEATURE_SIZE = 32 * 32 * 3
        self.CLIENT_EPOCHS_PER_ROUND = 1
        self.NUM_CLIENT_DATA = 500
        self.CLIENT_GAUSSIAN_STATE = False
        self.FEDERATED_OPTIMIZATION_STATE = False
        self.PERTURBATION_STATE = False
        self.TRAIN_STATE = False
        self.TEST_STATE = False
        self.CIFAR_10_STATE = False  # not right now
        self.CIFAR_100_STATE = True

federated_metrics = FederatedMetrics()

def create_table(header: list, csv):
    # compare federated strategies by their robustness, nominal metrics
    pass

def create_plot(x_axis, y_axis, x_data: list, y_data: list):
    # nominal metrics
    # federated metrics
        # communication rounds against test accuracy
        # communication rounds against train loss
        # communication rounds against changing client learning rate (possibly)

    # robustness metrics
        # epsilon vs l^2-bounded norm adversary; changing l-inf
        # epsilon vs l-infinity-bounded norm adversary; changing l-inf
        # measure for the change in loss sensitivity based on the increasing change of perturbation norm value (loss sensitivty vs epsilon norm values), again have both norm types defined as two lines
        # graph the l-2 norm and l-inf norm based on federated accuracy-under-attack (x-axis) and epsilon value (two lines)
        # (server-side trusted aggregator model comparison): client model set --> server-side model convergence/evaluation comparison against gaussian model and non-gaussian model (training with gaussian in training makes model fitted well to adversarial examples without overfitting, albeit slightly unclear how overfitting relates to fitting to more perturbed data)
        # FedAdagrad vs FedAvg comparison against hetereogeneous data (convergence, federated accuracy-under-attack)
        # dataset optimizations may be held constant for models with gaussian/non-gaussian changes

    # formal robustness metrics
        # specifications passed per configuration essentially (adversarial robustness properties satisfied)
        # specific variables within each formulation (comparison against each experiment setup)

    pass
