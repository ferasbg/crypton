# `src.deploy`
- Store utilities and configurations for running cloud instance. 
- Additionally will be responsible for storing sequential compute set for all classes and functions. 
- Will act as the class that will have end-to-end finite sequence of compute.


`src.deploy` acts as the main system runner e.g. development client. 

## Class Components
- `src.deploy.main`: store sequential runtime computations
- `src.deploy.deploy_utils`: store utilities to connect with all input functions from server nodes (e.g. `src.verification`, `src.prediction`, `src.crypto`, `src.adversarial`)
- `src.deploy.secure_nn`: link functions from `src.crypto` and `src.prediction.nn` to build privacy-preserving DCNN
- `src.deploy.specification_auth`: use industry-standard signature to authenticate specification computed on neural network at runtime
- `src.deploy.secure_verification`: link sub-class from `src.deploy.secure_nn` and `src.verification.specification`


## Components
- T1: Compute formal specification given safety property p and t and liveness properties, both of which are subsets of trace properties which are subsets of hyperproperties, defined on 3 requirements (cryptographic, adversarial, and the network). Setup `evaluate()` and run all models under privacy-preserving scheme and compute formal specifications on encrypted network to maintain cryptographic properties of system. Compute private inference and semantic image segmentation with deep convolutional neural network and formally prove trace properties of network.


