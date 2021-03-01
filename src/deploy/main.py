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
from verification.specification import RobustnessTrace, CheckTraceData
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
    # initialize the federated / dp privacy network architecture variant 
    crypto_network = CryptoNetwork()
    # network.train() # we need to dynamically pass in a dataset we want, the args should be dataset of already reshaped images and then its corresponding labels 
    # we need to iterate over all images in a specific dataset and use a specific model to evaluate all metrics for each respective model type (Network, CryptoNetwork)
    # initialize the perturbation_layer e.g. perturbation epsilon to apply to every input going through ImageDataGenerator, also note gaussian noise vector
    Adversarial.initialize_perturbation_layer()
    Adversarial.getGradients()
    Adversarial.setup_pixelwise_gaussian_noise(0.03, x_train[0])
    # initialize the l-norm bounded attack e.g. projected gradient descent attack (gradients compromised, maximize loss and inaccuracy), l^2 norm vs l-infinity norm for optimization after data augmentation 
    Adversarial.setup_pgd_attack(loss={})
    # initialize fgsm attack (e.g. use the gradients to maximize the loss e.g. inaccuracy of the classification with gradient sign method to generate adversarial example)
    Adversarial.setup_fgsm_attack(x_train[0], y_train[0], 0.3, model_parameters={}, loss={})

    # initialize the defined robustness specifications that are written as formal logical statements in sympy
    RobustnessTrace.adversarial_example_not_created_trace()
    RobustnessTrace.brightness_perturbation_norm_trace()
    # will execute projected gradient descent attack to use the gradients to maximize the loss with signed gradient method, fgsm attack to perturb inputs before they are passed to keras.layers.Input layer
    RobustnessTrace.fgsm_attack_trace()
    RobustnessTrace.smt_solver_trace_constraint()
    RobustnessTrace.pgd_attack_trace()
    RobustnessTrace.robustness_bound_check()

    # initialize the bounded model checker from verification.main.BoundedNetworkSolver that will evaluate and compare the network state (e.g. output layer vector norm) with respect to the input_layer given perturbation norm applied before input layer (ok, why not use autoencoders? write in paper)
    model_checker = BoundedNetworkSolver()
    # model checker for crypto_network
    crypto_model_checker = BoundedCryptoNetworkSolver() 

    # get reachable states for SMT Solvers
    # pass the reachable states or "worlds" e.g. output states that satisfy specification within some bound to satisfy the specification
    # create a list of reachable states of interest for specification and to pass to smt_solver_trace_check()
    model_checker.smt_solver_trace_check()
    model_checker.bmc_to_propositional_satisfiability()
    model_checker.get_relevant_reachable_states(network)
    model_checker.initialize_constraint_satisfaction_formula()
    model_checker.symbolically_encode_network(network)
    model_checker.traverse_robust_network_state_transitions()

    # return bmc_specification_status 
    crypto_model_checker.smt_solver_trace_check()
    crypto_model_checker.bmc_to_propositional_satisfiability()
    crypto_model_checker.get_relevant_reachable_states(network)
    crypto_model_checker.initialize_constraint_satisfaction_formula()
    crypto_model_checker.symbolically_encode_network(network)
    crypto_model_checker.traverse_robust_network_state_transitions()


    # return certified metrics
    Metrics.get_certified_metrics_for_network()
    Metrics.get_certified_metrics_for_crypto_network()
    # return crypto metrics
    Metrics.getCryptoMetrics()
    # return adversarial metrics
    Metrics.getNominalAdversarialMetrics()
    Metrics.getCryptoAdversarialMetrics()
    # return natural / nominal metrics
    Metrics.getNominalMetrics()


if __name__ == '__main__':
    main()
    # every module (e.g. src/verification/crypto) has it's own main and static functions for arbitrary deviation from sequential program execution
