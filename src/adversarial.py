import sympy
import numpy
import scipy
import os, sys
import tensorflow as tf
import cleverhans
import adversarial-robustness-toolbox as art

from model import Network

# applying image perturbations isn't in a layer, but rather before data is processed into the InputSpec and Input layer of the tff model


def compute_norm_bounded_perturbation(input_image, norm_type, perturbation_epsilon):
    '''Reuse for each additive perturbation type for all norm-bounded variants. Note that this is a norm-bounded ball e.g. additive and iterative perturbation attack.'''
    if (norm_type == 'l-inf'):
        for pixel in input_image:
            pixel*=perturbation_epsilon + 0.5 # additive perturbation given vector norm
    elif (norm_type == 'l-2'):
        for pixel in input_image:
            pixel*=perturbation_epsilon + 0.5

def brightness_perturbation_norm(input_image):
    sigma = 0.085
    brightness_threshold = 1 - sigma
    input_image = tf.math.scalar_mul(brightness_threshold, input_image)
    return input_image
