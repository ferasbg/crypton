import sympy
import numpy
import scipy
import os, sys
import tensorflow as tf
import cleverhans

class Adversarial():
    perturbation_epsilon: [0.1, 0.2, 0.3, 0.4, 0.5] # number of perturbed pixels (not gaussian noise)
    gradient_signed_method = {}
    network_gradients = {}

    @staticmethod
    def create_adversarial_example():
        # create adversarial example with perturbations to input_image iterating over entire train set to then pass into perturbed_train_generator
        raise NotImplementedError

    @staticmethod
    def setup_pixelwise_gaussian_noise(epsilon, input_image):
        # setup perturbation layer in network as adversarial defense/attack and perhaps use as utility function
        raise NotImplementedError

    @staticmethod
    def projected_gradient_descent_attack(layer_gradient, network_gradients):
        # converge to max loss to create adversarial example
        # to get layer_gradient: tape.gradient(loss, linear_layer.trainable_weights)
        raise NotImplementedError

    @staticmethod
    def fgsm_attack():
        raise NotImplementedError

    @staticmethod
    def create_adversarial_polytope():
        raise NotImplementedError

    @staticmethod
    def perturb_input_image(input_image):
        raise NotImplementedError