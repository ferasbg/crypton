import sys
import os
import time
import math
import torch
import argparse
import h5py
import json

"""
Define graphs and federated evaluation metrics.

Recorded metrics:

Natural Accuracy for Plaintext Network Trained With Geforce K80:
1563/1563 [==============================] - 9s 6ms/step - loss: 0.1419 - accuracy: 0.9522 - val_loss: 1.2321 - val_accuracy: 0.7442

Certified Accuracy Trained With Geforce K80:
1563/1563 [==============================] - {}s {}ms/step - certified_loss: 0.1419 - certified_accuracy: {} - certified_val_loss: {} - certified_val_accuracy: {}

Accuracy-Under-MPC Trained with Geforce K80:

Certified CryptoNetwork (Accuracy-Under-Federated/DP) Accuracy Trained With Geforce K80:

Graphs:
- epsilon vs l^2-bounded norm adversary
- epsilon vs l-infinity-bounded norm adversary
- compare accuracies of networks (note that adversarial examples optimized performance vs any other technique for natural training)
- BoundedNetworkSolver termination_time
- properties checked and the computation time for all trace properties
- PGD accuracy, FGSM accuracy
"""

class Metrics():
    '''
    Functions to compute natural/nominal/convential evaluation metrics, crypto-specific metrics, verification/certification metrics, and robustness metrics given adversarial attacks to network variant (public/crypto) under the state of being verified with certification checkers and defined formal trace properties.
    '''
    @staticmethod
    def getEpsilonNormAdversaryGraph(epoch_set, norm_set):
        raise NotImplementedError

    @staticmethod
    def perturbation_type_and_accuracy_under_perturbation():
        raise NotImplementedError
