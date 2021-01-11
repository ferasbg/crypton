# Crypton
Crypton: Formal Verification for Secure, Privacy-Preserving Deep Convolutional Neural Networks

## Overview
The purpose of this work is to build a system that implements signal temporal logic (STL) to formally verify the adversarial robustness with interval bound propagation of a deep convolutional neural network for semantic image segmentation in a privacy-preserving, zero-trust approach with respect to computational efficiency.



## System Components
- `crypton.src.Prediction`: Store the deep convolutional neural network for semantic image segmentation.
- `crypton.src.Verification`: Store neural network formal specification & verification algorithms, such as signal-temporal logic (STL).
- `crypton.src.Adversarial`: Store algorithms for automated input rectification, interval bound propagation, to compute adversarial robustness.
- `crypton.src.Crypto`: Store components to setup the SMPC scheme for the DCNN.
- `crypton.src.Analytics`: Compute statistical significance of all components of crypton.



