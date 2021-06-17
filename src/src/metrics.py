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
        Notes:
            - federated setting involves parent and child models e.g. server-side trusted aggregator and client-side models that are distillations of the parent model; understanding adv. regularization and fed. optimization to improve the server-side model as much as possible (which IS formalized through its adv. robustness) is crucial for a robust federated ML system in general 

        Todo:

            - Nominal Robustness: loss sensitivity (sigma), empirical robustness (unknown), test error, kl-divergence (mu), pristine/natural federated accuracy (alpha), federated accuracy-under-attack, perturbation norm (p_adv-e), time_per_round
            - Formal Robustness: Admissibility Constraint, Distance Constraint, Target Behavior Constraint, Perturbation Minimization, Loss Maximization, Allowed L-Inf Perturbations (Maximization Problem, possibly)

            - task: Update the state of variables for particular computations that can be graphed and stored into a table of use for the whitepaper.
            - note: formalizations are meant to assert or certify specific adversarial robustness properties of the client models and the trusted aggregator model (server model), so we'd fit these same equations in the context of client networks and client-specific hyper-parameters etc (L = (theta, x,y)), and the value is in fitting the optimization formulations of adversarial robustness to the federated setup (as stated many times before)
        
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
        # relate the client adv. and base loss and eventually the server loss after federated averaging (and secure aggregation) are computed
    

    # robustness metrics
        # grad norm type and increasing epsilon comparison graph
            # task: epsilon vs l^2-bounded norm adversary; changing l-inf
            # task: epsilon vs l-infinity-bounded norm adversary; changing l-inf
        # task: measure for the change in loss sensitivity based on the increasing change of perturbation norm value (loss sensitivty vs epsilon norm values), again have both norm types defined as two lines
        # task: graph the l-2 norm and l-inf norm based on federated accuracy-under-attack (x-axis) and epsilon value (two lines)
        # note: (server-side trusted aggregator model comparison): client model set --> server-side model convergence/evaluation comparison against gaussian model and non-gaussian model (training with gaussian in training makes model fitted well to adversarial examples without overfitting, albeit slightly unclear how overfitting relates to fitting to more perturbed data)
        # task: FedAdagrad vs FedAvg comparison against hetereogeneous data (convergence, federated accuracy-under-attack)
        # note: dataset optimizations may be held constant for models with gaussian/non-gaussian changes
        # task: lowest perturbation norm that doesn't damage the model (measure increasing norms against accuracy-under-attack for both adv. regularized model and base model, note the diffs)
        # task: measure for what models adapt the best to an increasing perturbation grad norm
        # task: formalize robustness at the client-level and at the federated strategy level along with the server-model level
        # note: We are executing a distributed perturbation / model spoofing (threat model) across a distributed set of client models with heterogeneous, corrupted data and other real-world scenarios regarding limited and unreliable data. Note how behavior changes as perturbation norm increases and distance between its max l-2/l-inf radius decreases (as epsilon goes up).
    
    # adversarial metrics
        # we will need to test whether perturbations both satisfy the optimization formulation as it increases per iteration, since we will use a set of norm-bounded perturbations and will assess what perturbations are "allowed" e.g. satisfy specification of formulation
        # we know the perturbation element (a subset of all the allowed and unallowed norms around the l-2/l-inf radius) will try to maximize the loss when the goal is to minimize loss during evaluation 
        # compare against comparable projects that are missing specific features e.g. federated adversarial regularization and adversarial + federated optimization
        # a note: the idea is to relate the effects of how input data is processed e.g. adversarial example and model architecture (GaussianNoise)
        # the goal is to see if simulating perturbation-like distortions/corruptions to the data (to simulate data (packet) loss, data heterogeneity, etc) to avoid adversarial model overfitting during its training iterations may help it optimize its norm-bounded robustness to an adversarial example during evaluation. We simulate real-world conditions in a federated setting, and measure for how we can generate adversarial examples that federated models can [satisfy the specification defined by the formalizations of the adversarial robustness property set.]
        # apply constant set of configs (perturbation in eval and increasing norm every round --> eps --> batch, gaussian noise, adv. regularization)
        # As mentioned earlier, if we are to assume more robust production ML systems, we also want to account for real-world scenarios involving the sparsity of data per client, corrupted data per client, compression problems in input data, threat models regarding model "spoofing" from an attacker, we need to have more clear "interpretability" or understanding of what to expect at even the scale of many client models. Later we may even fit these uncertainties to a rule set that depicts real-world data, as mentioned in this paper.
        # research innovation is in applying adversarial regularization to a novel federated setting where robust adversarial examples operate on adversarially regularized client models, and we also want to optimize the federated strategy used to update the server-side trusted aggregator model.


    # formal robustness metrics
        # specifications passed per configuration essentially (adversarial robustness properties satisfied)
        # specific variables within each formulation (comparison against each experiment setup)
        # relating adversarial input generation (variables) to certified loss (e.g. loss under attack and certification e.g. computing formal_robustness)

    pass
