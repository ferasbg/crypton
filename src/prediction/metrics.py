import sys
import os
import time
import math
import torch
import argparse
import h5py
import json

def compute_mi_fgsm(adversarial_sample_size, correct_adversarial_labels):
    raise NotImplementedError

def top_1():
    raise NotImplementedError

def top_5():
    raise NotImplementedError


def cross_entropy_loss():
    """ Compute Cross-Entropy Loss Given compute_softmax(Model model) """
    raise NotImplementedError

def adjusted_rand_index(true_mask, pred_mask, name='ari_score'):
    """Compute Adjusted Rand Score (ARI) which is a clustering similarity score.

    Args:

    Returns:

    Raises:


    References:


    """
    raise NotImplementedError


def intersection_over_union():
    raise NotImplementedError

def frequency_weighted_iou():
    raise NotImplementedError

def mean_iou():
    raise NotImplementedError

def boundary_f1_score():
    raise NotImplementedError

def global_average_accuracy():
    raise NotImplementedError

def class_average_accuracy():
    raise NotImplementedError


def mean_pixel_accuracy(correct_pixels, total_pixels):
    """Mean Pixel Accuracy. Proportion of all correctly classified pixels over the total pixels of the image, then compute sum of all proportions, then average it out.


    Args:

    Returns:

    Raises:


    References:


    """
    raise NotImplementedError


def overall_pixel(correct_pixels, total_pixels):
    """ Overall Pixel (OP) Accuracy: Proportion of Correctly Labeled Pixels to Total Pixels. Drawbacks include bias and overt generalization. """
    result = correct_pixels/total_pixels
    return result


def per_class():
    """Get Per-Class (PC) Accuracy for Correctly Labeled Pixels for Each Class, then Average Over Total Classes"""
    raise NotImplementedError
