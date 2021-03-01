import sys
import os
import time
import math
import torch
import argparse
import h5py
import json

"""
Define conventional/nominal, certification/verification (robustness, safety), mpc, adversarial evaluation metrics for network.

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

"""

class Metrics():
    '''
    Functions to compute natural/nominal/convential evaluation metrics, crypto-specific metrics, verification/certification metrics, and robustness metrics given adversarial attacks to network variant (public/crypto) under the state of being verified with certification checkers and defined formal trace properties.
    '''
    @staticmethod
    def natural_accuracy(correct_images, dataset):
        dataset_length = len(dataset)
        natural_accuracy = correct_images // dataset_length
        return natural_accuracy        

    @staticmethod
    def certified_robustness_accuracy(prediction_robustness_threshold, certified_robustness_norm):
        """robustSize(scores, , δ, L) returns the certified robustness size, which is then compared to the prediction robustness threshold T."""
        if (certified_robustness_norm > prediction_robustness_threshold):
            euclidean_distance = certified_robustness_norm - prediction_robustness_threshold
            return euclidean_distance
        else:
            return "Failed to get proper variables for robustness region, threshold, and robustness norm computed from specification given convergence/optimization setup."


    @staticmethod
    def nominal_kl_divergence():
        raise NotImplementedError

    @staticmethod
    def crypto_certification_accuracy():
        raise NotImplementedError

    @staticmethod
    def k_anonymity():
        """
        "this approach was proposed by Sweeney in 2002 [6]. A dataset is said to be k-anonymous if
        every combination of identity-revealing characteristics occurs in at least k different rows of the dataset.
        This anonymization approach is vulnerable to such attacks as background knowledge attacks."
        """
        raise NotImplementedError

    @staticmethod
    def get_certified_accuracy():
        raise NotImplementedError

    @staticmethod
    def get_certified_robustness_region():
        raise NotImplementedError

    @staticmethod
    def get_crypto_certified_robustness_region():
        raise NotImplementedError

    @staticmethod
    def get_certified_metrics_for_network():
        '''Batch return all the metrics for network given specifications are checked.'''
        print("Certified accuracy: {}".format(Metrics.get_certified_accuracy()))
        print("Certified robustness region: {}".format(Metrics.get_certified_robustness_region()))    

    @staticmethod
    def get_certified_metrics_for_crypto_network():
        '''Batch return all the metrics for crypto_network given specifications are checked.'''
        print("CryptoNetwork certified accuracy: {}".format(Metrics.crypto_certification_accuracy()))
        print("CryptoNetwork certified robustness region: {}".format(Metrics.get_crypto_certified_robustness_region()))    

    @staticmethod
    def getCryptoMetrics():
        '''Batch return all the crypto metrics for crypto_network.'''
        print("K-Anonymity: {}".format(Metrics.k_anonymity()))


    @staticmethod
    def getNominalAdversarialMetrics():
        raise NotImplementedError
    
    @staticmethod
    def getCryptoAdversarialMetrics():
        raise NotImplementedError
    
    @staticmethod
    def getNominalMetrics():
        raise NotImplementedError
    