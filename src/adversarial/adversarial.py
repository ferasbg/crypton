import sympy
import numpy
import scipy
import os, sys
import tensorflow as tf
import cleverhans

class Adversarial():
    @staticmethod
    def create_adversarial_example():
        # create adversarial example with perturbations to input_image iterating over entire train set to then pass into perturbed_train_generator
        raise NotImplementedError


    def setup_pixelwise_gaussian_noise(epsilon, input_image):
        # setup perturbation layer in network as adversarial defense/attack and perhaps use as utility function
        for iter in range(input_image.length):
            input_image[0][0]*=epsilon


    @staticmethod
    def setup_robustness_bound():
        raise NotImplementedError

    @staticmethod
    def setup_robustness_property():
        raise NotImplementedError

    @staticmethod
    def create_abstract_domain():
        raise NotImplementedError


class PGD():
    raise NotImplementedError

