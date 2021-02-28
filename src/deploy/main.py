import os
import sys
import tensorflow as tf
import random

from crypto.mpc_network import MPCNetwork

def main():
    # initialize the perturbation_layer e.g. perturbation epsilon to apply to every input going through ImageDataGenerator, also note gaussian noise vector
    
    # initialize fgsm attack (e.g. use the gradients to maximize the loss e.g. inaccuracy of the classification with gradient sign method to generate adversarial example)
    # initialize the l-norm bounded attack e.g. projected gradient descent attack (gradients compromised, maximize loss and inaccuracy), l^2 norm vs l-infinity norm for optimization after data augmentation 
    # initialize the defined robustness specifications that are written as formal logical statements in sympy
    # initialize the bounded model checker from verification.main.BoundedNetworkSolver that will evaluate and compare the network state (e.g. output layer vector norm) with respect to the input_layer given perturbation norm applied before input layer (ok, why not use autoencoders? write in paper)
    # during runtime after training model, get robustness certification accuracy, get robustness_region, get nominal_accuracy given adjustment in perturbation_epsilon e.g. [0.01, 0.03, 0.05]
    # pass input_params to model checker that is initialized with the object that represents the network's state-transitions to worry about and evaluate e.g. compare input-output vector norms, check against specification to verify trace property with satisfiability : this method is to reduce search complexity and to approach from more symbolic representation of the network e.g. composite function / computational-graph
    # compute formal specifications and return bmc_specification_status and abstract_network_specification_status
    # train model over cifar-10, and cifar-100 if necessary
    # return evaluation metrics are required for certification for a.i. and b.m.c 
    # initialize the public network, and then execute precedence properties (check pretrained trainable weights, check layers / type def / params for size, preprocess cifar-10 data)
    # initialize the mpc network architecture variant 
    # sequentially compute non-plaintext model given shares made up from required variables (keras.layers.Layer) to then compute the secret iterating over each layer during forward propagation for secure training
    # compute if the network's abstract output layer would be inside the robustness region e.g. satisfies robustness specification
    # compute adversarial training given compartmentalized dataset of cifar-10 images with initialized threat models
    # use mpc_network to train network for nominal_accuracy and certification_accuracy for the BoundedNetworkSolver 
    # for evaluation metrics, keep in mind that we want to compare mpc_network and public network for BoundedNetworkSolver and BoundedMPCNetworkSolver respectively to compare computational efficiency, public_network directly for abstract interpretation for certification_accuracy given perturbation epsilon list to iterate over and probably allocate some images for : to simplify, use a small number of images to test with each time that don't repeat
    # for each adversarial_attack which should be defined as functions part of the adversarial.adversarial.Adversarial class, simply execute the instance of each function and pass in the network and its network state variables as params 
    # will execute projected gradient descent attack to use the gradients to maximize the loss with signed gradient method, fgsm attack to perturb inputs before they are passed to keras.layers.Input layer
    # BoundedMPCNetworkSolver(MPCNetwork.layer)
    # return evaluation metrics

    raise NotImplementedError


if __name__ == '__main__':
    main()
    # every module (e.g. src/verification/crypto) has it's own main and static functions for arbitrary deviation from sequential program execution