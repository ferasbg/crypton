import os
import random
import sys
import numpy
import sympy
import scipy

import tensorflow as tf
import keras
from keras.datasets import cifar10
from crypto.crypto_network import CryptoNetwork
from metrics import Metrics
from model import Network 
import tensorflow_federated as tff 
import tensorflow_privacy as tpp 

def main():
    """
        - Note, we are initializing our constraint-satisfaction problem, our prepositional modal logic formula to assert robust output states after network state-transition T(x) for P(x,y) => Q(x,y) | x := y or input-output states match and are thus robust and thus correctness is proven. Negate non-robust output states and consider them in violation of the constraint specified (any thoughts on how you'd label an "output state" as a "violation"? What is "non-robust" anyway to you?).        
        - Note that this logic proves partial program correctness, since it's evaluating the input-output relations under the constraint of an adversarial attack on the network's inputs.
        - Convert convolutional network state into propositional formula given the constraint-satisfaction problem defined in trace.
        - Store some static variable state to indicate satisfiability in order to update the respective trace state to verify against specification given model checker.
        - Synthesize logical formula translated through encoding convolutional network as a constraint-satisfaction problem with respect to pre-condition and post-condition after network state-transition e.g. forwardpropagation. 
        - Note, the implication is satisfied given the network_postcondition e.g. output_state given perturbation norm and input_image as network_precondition
        - Note that x := input_class and y := output_class
        - Formally written as: Network ⊢ SAT ⟺ P(x,y) ⇒ Q(x,y) | P | ∀ x ∧ ∀ y     
        - why do we return this formula and why do we verify examples not created
        - use general formulations for each client network, its respective layers/components/functions with respect to the attack and its communication to other client networks and the server model
        - how do we "verify" robustness? By confirming accuracy through some optimization problem we formulate our model into? And then say oh look it is still functional?
        - Verify robustness given brightness perturbation norm-bounded attack against each input_image passed to network. 
        - Verify robustness given l-norm (l^2, l-infinity, l-1) bounded perturbation attack against each input_image passed to network.
        - Model is considered robust given that under the pre-condition of fast sign gradient method attack, the model maintains the post-condition of correctness.

        - Paper Name: "Federated Adversarial Neural Network Optimization"
        - Every adversarial attack invokes some form of domain adaptation (not necessarily) fitted to a perturbed input, and it optimizes the network by helping it converge over time. A federated setting is meant to assess any variance given attacks to a model that is on a device, simulating third-party attacks or even the noise associated with heterogeneous data that has a great distribution.
        - If we sequentially process different attacks, how do they affect the performance of the model and how do they relate to each other? How would we measure for that?
        - Compute adversarial attack (e.g. for adv_attack in adv_attacks) to perturb image_set with perturbation (δ) or attack P for each client k in a set of clients K, where the binary matrix of input_image x_i for the summation of xi and the dot product with the perturbation epsilon and a constant linear value. Compute this attack for each image for all images in the image_set (for image in image_set), and iterate attack for all clients, and then compute federated evaluation while passing perturbed inputs to tff-wrapped keras network. Compute federated evaluation metrics, and evaluate with respect to each round for each client, so iterating over the given NUM_EPOCHS and using BATCH_SIZE for each epoch. 
        - well we are creating an adversarial sample, but I think the question is measuring for how much it helps the model, when it is not helping the model, and why this happens
            - we measure for these changes given the precondition of security, and the use of this project is just so that we can have adaptive stress testing for ml models under federated production environments that handle dynamical and statistically noisy datasets

        - i am unsure on what the threshold implies given the optimization problem
            - my guess is that you'd have to see how other use adversarial attacks as optimization
            - we stress test our models with the assumption that they are secure, and the perturbed data is meant to model real-world data that is noisy

        - meta, but: in main() we define what'd be normally the sub-class of the FedBase class; functions must be abstracted from federated when used in main
        - the counter-example in this scenario is if we'd model the problem in a way that we'd search safe and unsafe states of a neural network, but that might not be the most effective method
            - we can downsize/distill the model, evaluate to a smaller hyper-parameter model, and use optimization as a way to simulate and understand what to expect if we use larger models (1T+ params, etc) 

        - l-infinity or l-p implies that p is a non-zero integer that indicates the vector-norm as if it'd be a ball in space
        - it seems that the variables below measure for whether an attack took place and if the model gave a correct result, but this doesn't help anyone
        - it's great that the attacks themselves are different, but it'd be nice to measure for how they are interdependent to each other in helping converge the model itself using stats/math
        - it's worthy to note how the difference in the adversarial attack also affects how the model would react to the next attack
            - if I use random norms to manipulate my image very granularly, how does the model react to if I'd instead change how I perturb the images based on a statistical distribution meant to counter-act the methods of backpropagation

        - I believe PGD and FGSM are both gradient-based attacks that are meant to simulate what'd happen if gradients are stolen (and under a federated scenario), so this is a feature regarding fallback and simply measuring for robustness and ofc the variables earlier that'd measure the change in the network's "probabilistic" accuracy given the image itself
        - It may be futile to use model checkers for models with 600m+ hyper-params, but it may be useful to model the problem differently even for a distilled model at the current scale
            - perhaps the better question to ask is how we setup a production-like ML environment and thrive in chaos (and adversarial attacks can invoke chaos, and we measure that chaos, and hone in how we can optimize our network with chaos)

        - now to confirm the process of "secure aggregation" and "federated averaging", I believe it's more a privacy-specific method to do the same gradient accumulation/update to the model as we'd do before, except not continuously, in the name of privacy standards.
        - is it worth measuring for stochastic communication for client-server model interaction given adversarial attack?
        - given federated env (secure agg, fedSGD), how does that change things up as far as formulating neural network behavior?
        - note that accuracy is the average accuracy of all the clients PER round
        - metrics to track each round: federated_accuracy_under_pgd, federated_accuracy_under_fgsm, federated_accuracy_under_norm_perturbation_type (np.linf, l2 norm)
        - partitioning dataset for different tests.
        - compute adversarial attack over all images for each client    
        - there's 10 clients, and cifar_train stores 500,000 examples, and cifar_test stores 100,000 examples.  
        - we want to split the test dataset for each client such that there's 5 partitions per client test_set, so if each client gets 10,000 images, and there are 5 adversarial attacks used, then there must be 2,000 test images per adversarial attack. 
        - track states for robustness specifications
        
        for client in clients:
            for image in image_set:
                adv_attack(image)

        - iterate epochs, batch_size over each round iterating over all the clients by default, then update server_state given model in server acting as aggregator
        for round_iter in range(NUM_ROUNDS):
            - initialize_fn initializes the tff server-to-client computation and next_fn updates the model in the server with the average of the clients' gradients updated from the training iteration / federated eval
            server_state, metrics = iterative_process(initialize_fn, next_fn)
            evaluate(server_state)
            print("Metrics: {}".format(metrics))

        The tf.data.Datasets returned by tff.simulation.ClientData.create_tf_dataset_for_client will yield collections.OrderedDict objects at each iteration, with the following keys and values:

        'coarse_label': a tf.Tensor with dtype=tf.int64 and shape [1] that corresponds to the coarse label of the associated image. Labels are in the range [0-19].
        'image': a tf.Tensor with dtype=tf.uint8 and shape [32, 32, 3], corresponding to the pixels of the handwritten digit, with values in the range [0, 255].
        'label': a tf.Tensor with dtype=tf.int64 and shape [1], the class label of the corresponding image. Labels are in the range [0-99].    
        
        - diff privacy for a federated network is dual privacy method it seems, are you still measuring for robustness the same way but factoring in attack vector variables and privacy scheme?
        - diff privacy IS gaussian noise (e.g. noise a data given distribution of rgb values and their vector norm values)
        - initialize dataset for each client, and for each perturbation_attack
        """"
  
    cifar_train, cifar_test = tff.simulation.datasets.cifar100.load_data()
