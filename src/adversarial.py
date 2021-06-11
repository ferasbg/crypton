import sympy
import numpy
import scipy
import os, sys
import tensorflow as tf
import cleverhans
import art

from model import Network

# tigher distribution
def get_larger_perturbation_epsilons():
    larger_perturbation_epsilons = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
    return larger_perturbation_epsilons

# "looser" distribution
def get_smaller_perturbation_epsilons():
    smaller_perturbation_epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return smaller_perturbation_epsilons

def compute_norm_bounded_perturbation(input_image, norm_type, perturbation_epsilon):
    '''Reuse for each additive perturbation type for all norm-bounded variants. Note that this is a norm-bounded ball e.g. additive and iterative perturbation attack.'''
    if (norm_type == 'l-inf'):
        for pixel in input_image:
            pixel*=perturbation_epsilon + 0.5 # additive perturbation given vector norm
    elif (norm_type == 'l-2'):
        for pixel in input_image:
            pixel*=perturbation_epsilon + 0.5

def brightness_perturbation_norm(input_image):
    # applying image perturbations isn't in a layer, but rather before data is processed into the InputSpec and Input layer of the tff model
    sigma = 0.085
    brightness_threshold = 1 - sigma
    input_image = tf.math.scalar_mul(brightness_threshold, input_image)
    return input_image
