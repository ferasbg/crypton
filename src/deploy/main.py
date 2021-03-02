import os
import random
import sys
import numpy
import sympy
import scipy

import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.datasets.cifar10 import load_data

from adversarial.main import Adversarial
from crypto.main import Crypto 
from crypto.crypto_network import CryptoNetwork
from nn.metrics import Metrics
from nn.network import Network 
from verification.specification import RobustnessTrace
from verification.main import VerifyTrace, BoundedCryptoNetworkSolver, BoundedNetworkSolver


def main():

    # initialize dataset and compartmentalize train/test/val for public/mpc/public_certify/mpc_certify
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # reshape into binary matrix
    x_train = x_train.reshape((-1, 32, 32, 3))
    x_test = x_test.reshape((-1, 32, 32, 3))

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # initialize public network, mpc network, adversarial attacks/defenses, and MPCProtocol
    network = Network()
    crypto_network = CryptoNetwork() # we need to initialize the federated eval given client generation (local models update global model) 

    Adversarial.setup_pixelwise_gaussian_noise(0.03, x_train[0])
    # initialize the l-norm bounded attack e.g. projected gradient descent attack (gradients compromised, maximize loss and inaccuracy), l^2 norm vs l-infinity norm for optimization after data augmentation 
    Adversarial.setup_pgd_attack(loss={})
    # initialize fgsm attack (e.g. use the gradients to maximize the loss e.g. inaccuracy of the classification with gradient sign method to generate adversarial example)
    Adversarial.setup_fgsm_attack(x_train[0], y_train[0], 0.3, model_parameters={}, loss={})

    '''
    # states to update 
    adversarial_sample_created = False
    robustness_threshold_state = False
    counterexample_verificationState = False
    lp_perturbation_status = False # l_p vector norm perturbation
    correctness_under_lp_perturbation_status = False
    brightness_perturbation_status = False
    correctness_under_brightness_perturbation = False # accuracy under brightness perturbation with (1-sigma) threshold
    fgsm_perturbation_attack_state = False
    fgsm_perturbation_correctness_state = False
    pgd_attack_state = False
    pgd_correctness_state = False
    smt_satisfiability_state = False

    '''

    # invoke traces in order to check the states after they're updated given the robustness analysis
    RobustnessTrace.adversarial_example_not_created_trace()
    RobustnessTrace.brightness_perturbation_norm_trace()
    RobustnessTrace.fgsm_attack_trace()
    RobustnessTrace.smt_constraint_satisfiability_trace()
    RobustnessTrace.pgd_attack_trace()

    model_checker = BoundedNetworkSolver()
    crypto_model_checker = BoundedCryptoNetworkSolver() 

    # evaluating if postconditions e.g. "worlds" or output states that satisfy specification within some bound to satisfy the specification
    model_checker.symbolically_encode_network(input_image={}, input_label={}) # perhaps iteratively given dataset of images and their labels
    model_checker.propositional_satisfiability_formula()


if __name__ == '__main__':
    main()
    # track PGD accuracy, FGSM accuracy (adversarial attacks), simply the accuracy computed given attack on input_images given we pass input_images and input_labels
