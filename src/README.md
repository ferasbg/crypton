# `crypton.src`
Crypton's core backend components and engine.

## Algorithms
- Algorithm 1: De-Encrypted Deep Convolutional Neural Network Training for Semantic Image Segmentation
- Algorithm 2: Encrypted Deep Convolutional Neural Network Training and Testing
- Algorithm 3: Design Formal Specifications Via Hyperproperties
- Algorithm 4: Compute Symbolic Interval Analysis & Reachability Analysis Given Bounded Deep Convolutional Neural Network for Robustness Trace Properties
- Algorithm 5: Compute Formal Specifications with Verification Techniques for Encrypted & Bounded Deep Convolutional Neural Network for Safety and Adversarial Robustness Verification


Note that the formal specifications are a representation of the formal constraints for the required properties of the neural network, and the methods to execute the property checks are done by the means of symbolic interval analysis, reachability analysis, signal temporal logic and property inference / checking, and bound propagation as a means of representing the networkstate and then setting up the verification / constraint-satisfaction problem for the property to be checked.



## Class Components
- `crypton.src.prediction`: store base class for DCNN
- `crypton.src.analytics`: store stats algorithms to compute metrics
- `crypton.src.monitor`: collect metrics at runtime for all components, render statistical significance with analytics class
- `crypton.src.verification`: store class for formal verification for DCNN and signal temporal logic
- `crypton.src.adversarial`: adversarial nodes for DCNN
- `crypton.src.client`: client to render analytics and processes
- `crypton.src.deploy`: setup instance to train and test network
- `crypton.src.crypto`: store algorithms for cryptographic scheme



## Requirements
- setup data preprocessing and network + eval/training, `iter in range(train_dataset.size())`, differentiate VGG16(), Sequential(), and Model(), track all args/params and member variables for training 
- train public network based on VGG-16 architecture, perhaps split dataset based on train/test/val for `Network`, `MPCNetwork`, and `Network` with Verification, and `MPCNetwork` with Verification. 
- setup mpc protocol to define crypto logic to encrypt `tf.keras.models.Input` tensor and dataset itself.
- setup formal specifications given written logic to compute the symbolic abstractions given the keras network state

