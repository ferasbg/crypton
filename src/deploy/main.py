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
from crypto.crypto_network import CryptoNetwork
from nn.metrics import Metrics
from nn.network import Network 
from verification.specification import RobustnessTrace # robustness specifications
from verification.main import VerifyTrace, BoundedCryptoNetworkSolver, BoundedNetworkSolver # verification methods
import tensorflow_federated as tff # tff computations
import tensorflow_privacy as tpp # diff privacy for noising input layer for client models

def main():

    '''
    
    Compute adversarial attack variant (e.g. for adv_attack in adv_attacks) to perturb image_set with perturbation (δ) or attack P for each client k in a set of clients K, where the binary matrix of input_image x_i for the summation of xi and the dot product with the perturbation epsilon and a constant linear value. Compute this attack for each image for all images in the image_set (for image in image_set), and iterate attack for all clients, and then compute federated evaluation while passing perturbed inputs to tff-wrapped keras network, and compute robustness specifications with abstraction-based verification defined with propositional / hoare logic for various robustness properties of the neural network. Compute federated evaluation metrics, and evaluate with respect to each round for each client, so iterating over the given NUM_EPOCHS and using BATCH_SIZE for each epoch. The average of the clients can be evaluated for the performance metrics.
    
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

    # note that accuracy is the average accuracy of all the clients PER round
    # metrics to track each round: federated_accuracy_under_pgd, federated_accuracy_under_fgsm, federated_accuracy_under_norm_perturbation_type (np.linf, l2 norm)
    # partitioning dataset for different tests.
    
    # compute adversarial attack over all images for each client    
    # there's 10 clients, and cifar_train stores 500,000 examples, and cifar_test stores 100,000 examples.  
    # we want to split the test dataset for each client such that there's 5 partitions per client test_set, so if each client gets 10,000 images, and there are 5 adversarial attacks used, then there must be 2,000 test images per adversarial attack. 
    # track states for robustness specifications
    
    for client in clients:
        for image in image_set:
            adv_attack(image)

    # iterate epochs, batch_size over each round iterating over all the clients by default, then update server_state given model in server acting as aggregator
    for round_iter in range(NUM_ROUNDS):
        # initialize_fn initializes the tff server-to-client computation and next_fn updates the model in the server with the average of the clients' gradients updated from the training iteration / federated eval
        server_state, metrics = iterative_process(initialize_fn, next_fn)
        evaluate(server_state)
        print("Metrics: {}".format(metrics))

    The tf.data.Datasets returned by tff.simulation.ClientData.create_tf_dataset_for_client will yield collections.OrderedDict objects at each iteration, with the following keys and values:

    'coarse_label': a tf.Tensor with dtype=tf.int64 and shape [1] that corresponds to the coarse label of the associated image. Labels are in the range [0-19].
    'image': a tf.Tensor with dtype=tf.uint8 and shape [32, 32, 3], corresponding to the pixels of the handwritten digit, with values in the range [0, 255].
    'label': a tf.Tensor with dtype=tf.int64 and shape [1], the class label of the corresponding image. Labels are in the range [0-99].    
    

    '''


    '''
    # initialize dataset for each client, and for each perturbation_attack
    cifar_train, cifar_test = tff.simulation.datasets.cifar100.load_data()

    crypto_network = CryptoNetwork()
    
    # diff datasets and crypto model checker checks wrapped keras model's state
    model_checker = BoundedNetworkSolver()
    crypto_model_checker = BoundedCryptoNetworkSolver() 

    # compute federated training for K clients using each adversarial attack, track perturbation and correctness states, and invoke model checker and robustness specifications 


    # compute tests given each adversarial attack


    # invoke traces in order to check the states after they're updated given the robustness analysis
    RobustnessTrace.adversarial_example_not_created_trace() # this is constant for each adversarial attack
    RobustnessTrace.brightness_perturbation_norm_trace()
    RobustnessTrace.fgsm_attack_trace()
    RobustnessTrace.smt_constraint_satisfiability_trace()
    RobustnessTrace.pgd_attack_trace()

    
    # evaluating if postconditions e.g. "worlds" or output states that satisfy specification within some bound to satisfy the specification
    model_checker.symbolically_encode_network(network_precondition={}, network_postcondition={}) # perhaps iteratively given dataset of images and their labels
    model_checker.propositional_satisfiability_formula()


if __name__ == '__main__':
    main()