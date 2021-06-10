import sympy
import numpy
import scipy
import os, sys
import tensorflow as tf
import cleverhans

from nn.network import Network
from nn.metrics import Metrics

# a "perturbation layer" IS-AN "adversarial layer", and it'd do a norm-bounded perturbation attack, else some form of a gradient attack or other model attack, anything to do "gradient ascent" or diverge away

# match the total epsilon values in the list to the number of attacks
# for attack in attacks:   
    # eps = perturbation_epsilon[attack]

perturbation_epsilon = [0.1, 0.2, 0.3, 0.4, 0.5] 
pgd_attack_state = False
fgsm_attack_state = False
l2_norm_perturbation_attack_state = False
l_infinity_norm_perturbation_attack_state = False
brightness_perturbation_norm_state = False

def compute_norm_bounded_perturbation(input_image, norm_type, perturbation_epsilon):
    '''Reuse for each additive perturbation type for all norm-bounded variants. Note that this is a norm-bounded ball e.g. additive and iterative perturbation attack.'''
    if (norm_type == 'l-inf'):
        for pixel in input_image:
            pixel*=perturbation_epsilon + 0.5 # additive perturbation given vector norm
    elif (norm_type == 'l-2'):
        for pixel in input_image:
            pixel*=perturbation_epsilon + 0.5

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
    pass


def create_adversarial_pattern(input_image, input_label):
    tape = tf.GradientTape()
    tape.watch(input_image) # map gradients
    prediction = (input_image) # given x to VGG-Net 
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    loss = loss_object(input_label, prediction) # total error
    input_image_label = []
    perturbations = create_adversarial_pattern(input_image, input_image_label)
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)

    return signed_grad


def compute_projected_gradient_descent_attack(conv2d_gradients, network_loss, l_infinity_norm=0.2, l2_norm=2.0):
    # converge to max loss to create adversarial example
    # to get layer_gradient: tape.gradient(loss, linear_layer.trainable_weights)
    pass

def brightness_perturbation_norm(input_image):
    sigma = 0.085
    brightness_threshold = 1 - sigma
    input_image = tf.math.scalar_mul(brightness_threshold, input_image)
    return input_image
