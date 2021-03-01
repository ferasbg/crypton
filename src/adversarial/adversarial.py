import sympy
import numpy
import scipy
import os, sys
import tensorflow as tf
import cleverhans

from nn.network import Network
from crypto.mpc_network import MPCNetwork
from crypto.mpc import MPC
from nn.metrics import Metrics

class Adversarial():
    gradient_signed_method = {}
    network_gradients = getGradients()
    perturbation_epsilon = 0.3 # perturbation epsilon

    @staticmethod
    def initialize_perturbation_layer():
        raise NotImplementedError

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
    def setup_pgd_attack():
        raise NotImplementedError

    @staticmethod
    def setup_fgsm_attack(input_image, input_image_label, perturbation_epsilon, model_parameters, loss):
        """Fast Gradient Signed Method (FGSM) attack as described in [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) by Goodfellow *et al*. This was one of the first and most popular attacks to fool a neural network.

        Note, we will compute a pixelwise norm perturbation with respect to the proportionality of the pixel to its corresponding loss value, in order to maximize the loss, e.g. making it inaccurate. It is given that model is pretrained and the model parameters are constant.
        
        Args:
        -   x : Original input image.
        -   y : Original input label.
        -   $\epsilon$ : Multiplier to ensure the perturbations are small.
        -   $\theta$ : Model parameters.
        -   $J$ : Loss.

        Returns: adv_x e.g. adversarial image with perturbations with respect to adversarial optimization.

        """
        perturbations = Adversarial.create_adversarial_pattern(input_image, input_image_label)
        for i in enumerate(1):
            adv_x = input_image + perturbation_epsilon*perturbations # matmul with perturbation epsilon
            adv_x = tf.clip_by_value(adv_x, -1, 1)
            i+=1
        
        return adv_x

    @staticmethod
    def create_adversarial_pattern(input_image, input_label):
        with tf.GradientTape() as tape:
            tape.watch(input_image) # map gradients  
            network = Network()
            prediction = (input_image) # given x to VGG-Net # rework model design
            loss_object = tf.keras.losses.CategoricalCrossentropy()
            loss = loss_object(input_label, prediction) # total error
        
        gradient = tape.gradient(loss, input_image)

        signed_grad = tf.sign(gradient)
        return signed_grad

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