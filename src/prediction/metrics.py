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


"""

class Metrics():
    def __init__(self):
        self.true_positive_pixels = 0
        self.false_positive_pixels = 0
        self.false_negative_pixels = 0
        self.misclassified_pixels = 0

        # acc
        self.mean_acc = 0

        # perturbation
        self.gaussian_epsilon = 0.10
        self.perturbation_epsilon = 0.05

    @staticmethod
    def top_1(self):
        raise NotImplementedError

    @staticmethod
    def nominal_cross_entropy_loss(self):
        """ Compute Cross-Entropy Loss Given compute_softmax(Model model) """
        raise NotImplementedError

    @staticmethod
    def boundary_f1_score(self):
        raise NotImplementedError

    @staticmethod
    def global_average_accuracy(self):
        raise NotImplementedError

    @staticmethod
    def class_average_accuracy(self):
        raise NotImplementedError

    @staticmethod
    def mean_pixel_accuracy(self, correct_pixels, total_pixels):
        """Mean Pixel Accuracy. Proportion of all correctly classified pixels over the total pixels of the image, then compute sum of all proportions, then average it out.
        """
        raise NotImplementedError

    @staticmethod
    def overall_pixel(self, correct_pixels, total_pixels):
        """ Overall Pixel (OP) Accuracy: Proportion of Correctly Labeled Pixels to Total Pixels. Drawbacks include bias and overt generalization. """
        result = correct_pixels/total_pixels
        return result

    @staticmethod
    def per_class(self):
        """Get Per-Class (PC) Accuracy for Correctly Labeled Pixels for Each Class, then Average Over Total Classes"""
        raise NotImplementedError


    @staticmethod
    def certified_robustness_accuracy(prediction_robustness_threshold, certified_accuracy):
        """robustSize(scores, , δ, L) returns the certified robustness size, which is then compared to the prediction robustness threshold T."""
        certified_robustness_size = {}
        if (certified_robustness_size > prediction_robustness_threshold):
            return certified_robustness_size

    @staticmethod
    def nominal_kl_divergence():
        raise NotImplementedError


    


