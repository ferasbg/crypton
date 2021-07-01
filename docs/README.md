# Adversarially Robust Federated Optimization with Neural Structured Learning
The purpose of this work is to certify the adversarial robustness of a server-side trusted aggregator model given the optimizations done both for adversarial regularization at the client-level and the use of adaptive federated optimization which can adapt to heterogeneous, corrupt, low-resolution, and sparse client-side image data, to simulate and thrive in real-world federated learning environments.

## Future Work
- create secure gRPC connection between client nodes and trusted aggregator parent node
- run with ray as a distributed local graph specified by model and task, to aggregate federated environments
- add support for a set of additional certification algorithms of formal robustness as optimization
- dynamically analyze/extrapolate patterns from the set of robustness certification algorithms used for the federated environment of client-server models
- write protocol that can generalize across different neural network architectures with respect to federated environment
- add differential privacy as a precondition to evaluating formal/provable guarantees of the server/client models, and federated strategy (confidentiality --> privacy)

