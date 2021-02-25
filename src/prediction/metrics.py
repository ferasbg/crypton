import sys
import os
import time
import math
import torch
import argparse
import h5py
import json


class Metrics():
    def __init__(self):
        # iou
        self.mean_iou = 0
        self.true_positive_pixels = 0
        self.false_positive_pixels = 0
        self.false_negative_pixels = 0
        self.misclassified_pixels = 0

        # acc
        self.mean_acc = 0
        self.pixelwise_acc = 0

        # perturbation
        self.gaussian_epsilon = 0.10
        self.perturbation_epsilon = 0.05

    @staticmethod
    def compute_mi_fgsm(self, adversarial_sample_size, correct_adversarial_labels):
        raise NotImplementedError

    @staticmethod
    def top_1(self):
        raise NotImplementedError

    @staticmethod
    def top_5(self):
        raise NotImplementedError

    @staticmethod
    def cross_entropy_loss(self):
        """ Compute Cross-Entropy Loss Given compute_softmax(Model model) """
        raise NotImplementedError

    @staticmethod
    def adjusted_rand_index(self, true_mask, pred_mask, name='ari_score'):
        """Compute Adjusted Rand Score (ARI) which is a clustering similarity score.
        """
        raise NotImplementedError


    @staticmethod
    def mean_intersection_over_union(self):
        raise NotImplementedError

    @staticmethod
    def frequency_weighted_iou():
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
