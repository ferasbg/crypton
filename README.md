# Crypton
Crypton: Formal Verification for Secure, Privacy-Preserving Deep Convolutional Neural Networks

## Overview
The purpose of this work is to build a system that implements signal temporal logic (STL) to formally verify the adversarial robustness with interval bound propagation of a deep convolutional neural network for semantic image segmentation in a privacy-preserving, zero-trust approach with respect to computational efficiency.



## System Components
- `crypton.src.prediction`: Store the deep convolutional neural network for semantic image segmentation.
- `crypton.src.verification`: Store neural network formal specification & verification algorithms, such as signal-temporal logic (STL).
- `crypton.src.adversarial`: Store algorithms for automated input rectification, interval bound propagation, to compute adversarial robustness.
- `crypton.src.crypto`: Store components to setup the SMPC scheme for the DCNN.
- `crypton.src.analytics`: Compute statistical significance of all components of crypton.
- `crypton.src.monitor`: data processor to send to `crypton.src.analytics`
- `crypton.src.deploy`: runner object to run instance to compute sequential computation  (formal specifications on encrypted DCNN)


## System Workflow
Compute formal specifications to certify the safety and robustness of the privacy-preserving deep convolutional neural network for semantic image segmentation that is trained through distributed workers that compute on shared secrets, more specifically the specified `.train()` sequential computation. 
