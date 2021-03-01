import sympy
import numpy
import scipy
import os, sys
import tensorflow as tf
import cleverhans

class Adversarial():
    gradient_signed_method = {}
    network_gradients = getGradients()
    perturbation_epsilon = 0.3 # perturbation epsilon


    @staticmethod
    def create_adversarial_example():
        # create adversarial example with perturbations to input_image iterating over entire train set to then pass into perturbed_train_generator
        raise NotImplementedError

    @staticmethod
    def setup_pixelwise_gaussian_noise(epsilon, input_image):
        # setup perturbation layer in network as adversarial defense/attack and perhaps use as utility function
        raise NotImplementedError

    @staticmethod
    def getGradients():
        """
       Computes the gradients of outputs w.r.t input image.

        Args:
            img_input: 4D image tensor
            top_pred_idx: Predicted label for the input image

        Returns:
            Gradients of the predictions w.r.t img_input

        images = tf.cast(img_input, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(images)
            preds = model(images)
            top_class = preds[:, top_pred_idx]

        grads = tape.gradient(top_class, images)
        return grads

        """
        raise NotImplementedError

    @staticmethod
    def projected_gradient_descent_attack(layer_gradient, network_gradients):
        # converge to max loss to create adversarial example
        # to get layer_gradient: tape.gradient(loss, linear_layer.trainable_weights)
        raise NotImplementedError

    @staticmethod
    def fgsm_attack(model_parameters, input_image, cost_function):
        """Fast-sign gradient method, denoted as η = sign (∇xJ(θ, x, y)). Generate adversarial examples to pass into network."""
        raise NotImplementedError

    @staticmethod
    def perturb_input_image(input_image):
        raise NotImplementedError

    @staticmethod
    def brightness_perturbation_norm(input_image):
        sigma = 0.085
        brightness_threshold = 1 - sigma
        input_image = tf.math.scalar_mul(brightness_threshold, input_image)