# ARCHITECTURE
Defined specs e.g. implementation given mathematical formulation

## Table of Contents
- [Core Components](#core-components)
- [Mathematical Formulation](#mathematical-formulation)

## Core Components

### `src.deploy`
- [main.py](#deploy-main)
### `src.verification`
- [specification.py](#)
- [main.py](#)
- [stl.py](#)
### `src.src.`
- [network.py](#)
### `src.crypto`    
- [mpc.py](#)
- [mpc_network.py](#)
### `src.adversarial`   
- [`src.adversarial`](#src-adversarial)


## `deploy.main`
-
-
-

## `src.src.network.Network`
-
-
-

## `src.crypto.mpc`
-
-
-

## `src.crypto.mpc_network`
-
-
-

## `verification.specification.RobustnessTrace`
-
-
-

## `verification.main.BoundedNetworkSolver`
-
-
-

## `verification.BoundedMPCNetworkSolver`
-
-
-

## `verification.stl`
-
-
-

## `src.adversarial.Adversarial`
-
-
-

## Mathematical Formulation


## NOTES
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


## Requirement
In each core component, define the specification for programming model and implementation, 2) the mathematical formulation of each class and function, so have a table for each class for each file in each component, and define the dependency in terms of input params and functions and classes, when they are used and how (e.g. client/entry point, finite-state abstraction, compute verification, check trace property, encrypt network)

## Future Work
- use abstract interpretation for an encrypted MPC network
- GAN for scaling unsupervised learning in terms of input data
- continuous monitoring / automated reasoning that can adapt to scaled network hyperparameters and layers and even ensembles

