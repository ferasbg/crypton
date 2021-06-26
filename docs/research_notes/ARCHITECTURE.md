# ARCHITECTURE
Defined specs e.g. implementation given mathematical formulation

## Overview
Train neural network with ds encrypted with dp, and train with mpc. Decrypt, and symbolically encode model into constraint-satisifaction problem to be proved by `BoundedNetworkSolver`, to then check against the robustness specifications 

## `nn.network.Network`
- Define the convolutional network, given it is a computational graph and composite function.
- Note that in this module, there is a `Metrics` class which will store all the evaluation metrics with respect to nominal/certification accuracy given perturbation_epsilon, mpc_network_accuracy and mpc_network_accuracy_under_certification (verifying mpc network), kl-divergence where necessary, and robustness properties that were checked and verified, satisfiability/non-satisfiability (either way, it proves that with neural network defenses, neural network is more robust and will be more so during testing) 

## `src.crypto.mpc`
- Define the multi-party computation functions required for training the convolutional network (`nn.network.Network`). Note that DP noises the gradients and metadata that can reconstruct the original information, and MPC encodes the noisy gradients such that it can be computed in secure setting where reconstruction process is adversarially robust.
- Evaluate the encryption and decryption process of the Tensors, Layers, network_metadata e.g. gradients 

## `src.crypto.mpc_network`
- apply mpc protocol to convolutional network layers or perhaps to inputs to `keras.layers.Input` layer, depending on current literature
- Setup local member `train()` function that can encrypt the network layers and setup the secrets and compartmentalize the shares to iteratively and sequentially compute over the network during forward propagation. Note evaluation metrics to evaluate security properties of network.

## `verification.specification.RobustnessTrace`
- Define the robustness property specifications that converge temporal properties and symbolic logical formulae in terms of the model checker analyzing the convolutional network during runtime. 
- robustness trace 1: given lp-norm perturbation, the `output_class` is under certain threshold p that indicates the minimum for robustness
- robustness trace 2a: given brightness perturbation (maximize pixel intensity to maximize loss), model can reach certain robustness threshold after a set of epochs (this is important), then it indicates robustness, if at any point it doesn't satisfy robustness specification, it was formally verified that adversarial norm perturbation attacks can distort the input-output relationships e.g. given input_image and output class corresponding to its label
- robustness trace 2b: given projected gradient descent attack meant to distort backpropagation process via compromised gradients to maximize loss, assert that model updates its convergence to local minima with gradient descent correctly given bounds
- robustness trace 3: network is not making misclassifications with respect to L-norm (infinity, l^2, l-1)
## `verification.main.BoundedNetworkSolver`
- Setup bounded model checking algorithm to evaluate reachable states and to encode network symbolically with finite-state abstraction in order to scale to larger param size of network.
- Evaluate input-output relationships so that it satisfies the trace property that represents the model's robustness.
- 
## `verification.BoundedMPCNetworkSolver`
- Utilize the MPC logic written in `MPC()` to then decrypt the network state to encode symbolically.
-


## TECHNICAL NOTES
- use `assert` vs conditional to check Sequential.layer_name type with `assert instanceof X`
- define static functions inside classes for correct sequential workflow
- many papers have used different number of layers to differ their architectures, as well as comparing against existing methods in terms of evaluation metrics so make sure to do that
- **iterate on symbolic, parameterized, bounded model checking for a conv neural network encrypted with smpc + dp** with finite-state symbolic representation of the model and evaluating reachable states that can certify that the model's output layer and perhaps hidden layer computations will be equal to or exceed the threshold of robustness required for correct classifications.
- can we use perturbation_epsilon and tf.linalg.matmul to modify the input_image?
- should we re-use eran code? Or is it better to understand what ERAN is doing and avoid reading other code and instead getting context to re-create
- since training & architecture mods differ given mpc protocol, how would we decrypt a network for abstract interpretation or just restrict this process at runtime with the `BoundedMPCNetworkSolver`.
- we can run dual networks (Network and MPCNetwork) possibly to assess difference in execution_time for model checker, since state-transitions are encrypted or perhaps gradients are encrypted and defined as shares
- ok, change of course. The clearest way to execute this would be through temporal properties and property inference with input-output vector state comparison vs. generating an abstraction and so forth, since I can remove abstract interpretation entirely and focus on model checking and temporal logic + property specifications
- better to focus on defining the core specification and certifying the specification being satisfied based on the output_layer or class given the adversarial example generation attempt with fgsm attack and pgd attack to maximize loss
- focusing on that central idea and then designing mpc_network to focus on the privacy component maintains the focus of the system where we encode a large network into a logical constraint problem, decrypt and reconstruct the network itself (given mpc_protocol), and then check at runtime with either mpc_operations or public network_operations, far simpler given decryption logic with respect to architecture vs. training e.g. secret sharing + sequential computation iterating over network N


## GENERAL NOTES
In each core component, define the specification for programming model and implementation, 2) the mathematical formulation of each class and function, so have a table for each class for each file in each component, and define the dependency in terms of input params and functions and classes, when they are used and how (e.g. client/entry point, finite-state abstraction, compute verification, check trace property, encrypt network)

## Future Work
- use abstract interpretation for an encrypted MPC network
- GAN for scaling unsupervised learning in terms of input data
- continuous monitoring / automated reasoning that can adapt to scaled network hyperparameters and layers and even ensembles

