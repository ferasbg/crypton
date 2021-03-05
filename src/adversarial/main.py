import sympy
import numpy
import scipy
import os, sys
import tensorflow as tf
import cleverhans

from nn.network import Network
from crypto.crypto_network import CryptoNetwork
from nn.metrics import Metrics


class Adversarial():
    '''
    Note that every adversarial attack function will compute on individual images, but we must perturb all images iterating over the entire image sets used given convention. To perturb images passed for each local model for federated averaging, it is very important to perturb the image such that the vectorized Tensor passed with crypto.federated.preprocess() is sufficient in terms of tensor shape, and that we perturb an image_set. Note that when we process the image_sets for our test_set, we will perturb all the images in the test_set, and THEN create clients to compute federated averaging under secure aggregation environment.
    '''
    perturbation_epsilon = 0.3 # perturbation epsilon is constant for each attack variant
    pgd_attack_state = False
    fgsm_attack_state = False
    l2_norm_perturbation_attack_state = False
    l_infinity_norm_perturbation_attack_state = False
    brightness_perturbation_norm_state = False


    @staticmethod
    def compute_norm_bounded_perturbation(input_image, norm_type, perturbation_epsilon):
        '''Reuse for each additive perturbation type for all norm-bounded variants. Note that this is a norm-bounded ball e.g. additive and iterative perturbation attack.'''
        if (norm_type == 'l-inf'):
            for pixel in input_image:
                pixel*=perturbation_epsilon + 0.5 # additive perturbation given vector norm
        elif (norm_type == 'l-2'):
            for pixel in input_image:
                pixel*=perturbation_epsilon + 0.5
          

    @staticmethod
    def compute_fgsm_attack(input_image, input_image_label, perturbation_epsilon, model_parameters, loss):
        """Fast Gradient Signed Method (FGSM) attack as described in [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) by Goodfellow *et al*. This was one of the first and most popular attacks to fool a neural network.

        Note, we will compute a pixelwise norm perturbation with respect to the proportionality of the pixel to its corresponding loss value, in order to maximize the loss, e.g. making it inaccurate. It is given that model is pretrained and the model parameters are constant.

        Formally denoted as η=ϵ sign(∇ₓ J(θ,x,y)).


        Perturbation Type: Non-Iterative

        "Consider the dot product between a weight vector w and an adversarial example x˜: w>x˜ = w>x + w>η. The adversarial perturbation causes the activation to grow by w>η.We can maximize this increase subject to the max norm constraint on η by assigning η = sign(w). If w has n dimensions and the average magnitude of an element of the weight vector is m, then the activation will grow by mn. Since ||η||∞ does not grow with the dimensionality of the problem but the change in activation caused by perturbation by η can grow linearly with n, then for high dimensional problems, we can make many infinitesimal changes to the input that add up to one large change to the output. We can think of this as a sort of “accidental steganography,” where a linear model is forced to attend exclusively to the signal that aligns most closely with its weights, even if multiple signals are present and other signals have much greater amplitude. This explanation shows that a simple linear model can have adversarial examples if its input has sufficient dimensionality. Previous explanations for adversarial examples invoked hypothesized properties of neural networks, such as their supposed highly non-linear nature. Our hypothesis based on linearity is simpler, and can also explain why softmax regression is vulnerable to adversarial examples."

        Args:
        -   x : Original input image.
        -   y : Original input label.
        -   $\epsilon$ : Multiplier to ensure the perturbations are small.
        -   $\theta$ : Model parameters.
        -   $J$ : Loss.

        Returns: adv_x e.g. adversarial image with perturbations with respect to adversarial optimization. The computational efficiency of computing gradient descent with backpropagation is not affected.

        References:
            - https://github.com/gongzhitaao/tensorflow-adversarial/blob/master/attacks/fast_gradient.py
            - https://neptune.ai/blog/adversarial-attacks-on-neural-networks-exploring-the-fast-gradient-sign-method
            - https://arxiv.org/pdf/1811.06492.pdf
            - https://deepai.org/publication/simultaneous-adversarial-training-learn-from-others-mistakes


        """
        perturbations = Adversarial.create_adversarial_pattern(input_image, input_image_label)
        for i in enumerate(1):
            adv_x = input_image + perturbation_epsilon*perturbations # matmul with perturbation epsilon
            adv_x = tf.clip_by_value(adv_x, -1, 1) # clip Tensor values between tuple [-1,1]
            i+=1

        return adv_x

    @staticmethod
    def create_adversarial_pattern(input_image, input_label):
        with tf.GradientTape() as tape:
            tape.watch(input_image) # map gradients
            prediction = (input_image) # given x to VGG-Net 
            loss_object = tf.keras.losses.CategoricalCrossentropy()
            loss = loss_object(input_label, prediction) # total error

        gradient = tape.gradient(loss, input_image)
        signed_grad = tf.sign(gradient)
        return signed_grad

    @staticmethod
    def compute_projected_gradient_descent_attack(conv2d_gradients, network_loss, l_infinity_norm=0.2, l2_norm=2.0):
        # converge to max loss to create adversarial example
        # to get layer_gradient: tape.gradient(loss, linear_layer.trainable_weights)
        raise NotImplementedError

    @staticmethod
    def brightness_perturbation_norm(input_image):
        sigma = 0.085
        brightness_threshold = 1 - sigma
        input_image = tf.math.scalar_mul(brightness_threshold, input_image)
        return input_image
