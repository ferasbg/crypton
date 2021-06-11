import sys
import os
import time
import math
import torch
import argparse
import h5py
import json

# this differs from client to client, and batch/episode per epoch per round
federated_accuracy_under_attack = 0
natural_accuracy = 0
federated_natural_accuracy = 0

# epsilon vs l^2-bounded norm adversary
# epsilon vs l-infinity-bounded norm adversary
# compare accuracies of networks (note that adversarial examples optimized performance vs any other technique for natural training)
# change in perturbation over iterations (episodes, rounds, epochs : diff iteration granularities)