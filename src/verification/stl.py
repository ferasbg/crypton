# -*- coding: utf-8 -*-
# Copyright 2021 Feras Baig

import os
import time
import random
import torch
from torch import nn



class STLSpecification(nn.Module):
    """ Compute Formal Specification Given Trace Properties"""

    """ initialize variables to analyze network and to compute bounds
    # setup trace properties t (safety properties of network e.g. network is computing logits in last layer under correct bounds, do not compute classification given defined adversarial perturbation)
    # function set_1: safety properties of network
    # function set_2: liveness properties of network: number of activated neurons surpasses certain threshold, all layers compute and update tensor matrices correctly to update model during training, model computes IBP to guarantee adversarial robustness crosses defined threshold, concurrency with various classes computes in proper sequential order, model layers are encrypted with SMPC scheme, robust against adversarial examples"""
    pass


