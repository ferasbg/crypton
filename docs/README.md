# Adversarially Robust Federated Optimization with Neural Structured Learning
The purpose of this work is to build an adversarially robust federated system by utilizing adaptive and non-adaptive federated optimization techniques, as well as optimizing the adversarial regularization techniques used at the client-level in order to build a robust trusted aggregator model. More specifically, we aim to use neural structured learning and adaptive federated optimization together to build an adversarially robust federated learning system that can adapt to heterogeneous, sparse, and perturbed data that would be faced in a production-level federated environment.

## Introduction
- There is a disconnect in terms of system design relating to federated systems for specifically image recognition and other computer vision tasks when it comes to performing well on real-world data. Models converge well on fixed, IID data with unrealistic quality, sparsity, fault tolerance, adaptivity, and so on, so this system is built to adapt to the data state and the system state (adaptive strategies, adversarially regularized client models). The goal for this paper is to integrate scalable optimization techniques both at the model level and at the system level that can utilize adaptivity and robustness against real-world data that has surface variations, perturbations, corruptions, and other natural distortions, not to mention its non-IID nature and sparsity. Our system can handle all of these environmental variables.
- This paper introduces a novel and scalable connection between adversarial regularization and neural structured learning for production-level federated learning.

## Problem Statement
- Formulation is in terms of system components and subcomponents. Strategy (adaptive, non-adaptive), adversarial regularization technique, and robustness specifications (checks) for server-side model evaluation. We can have a base formulation and extend it to various forms of formal "scrutiny" to ensure dependability/reliability under variability.
- Formalize the problem that is being solved in terms of the components responsible, neglecting unnecessarily specific variables that are default settings and thus preconditional assumptions (safe to assume, so extraneous in problem formulation).

## Crypton Architecture

<div align="center">
    <img src="https://github.com/ferasbg/crypton/blob/master/docs/media/nsl_architecture.png" width="800" align="center">
</div>

#### System Components
- `client.py`: define the client models and their configuration subcomponents.
- `server.py`:  
- `simulation.py`: aggregate server-client processes iteratively in terms of the experimental configuration permutations
#### System Subcomponents
- `HParams`: adversarial hyperparameter settings
- `AdvRegClient`: client that uses neural structured learning
- `Client`: client that uses nominal regularization techniques and can be configured for corruption learning

## Results and Evaluation
- Store results given each experiment configuration defined in `exp-config-run.sh` that will iteratively process train/test client partitions and client/model/server/strategy configurations to check what optimizations conform best to the specifications for the certification component.

## Research Innovations
- Implementing adversarially regularized deep convolutional networks in a federated setting, and using adaptive federated optimization. The purpose of which are to build a system that can adapt to the real-world environment of data sparsity, corruptions, and heterogeneity. 
- First implementation that formalizes neural structured learning algorithm as an adversarial regularization technique combined with other forms of regularization under federated setting with respect to adversarial robustness certifications for a server-side trusted aggregator model.
- Companies can use the base architecture into their own machine learning systems in terms of their individual components that involve singularly defined tasks in their end-to-end system, and define federated (cluster) environments for each of their components while maintaining scalability and robustness.

## Discussion
- Statements must either disprove, prove, certify, or validate the behavior, properties, and entropy (as implicit attitude in discussion without mathematical formalization of entropy itself other than the entropy method used) involved with the results that accounts for the default settings and experiment configs for each component.
- Reason about why particular behavior was observed.
- State the importance, validity, extensibility, reliability, and significance of the findings observed and reasoned about in the Discussion section.
- State the research innovations made on the paper, and how the information itself can be used (dynamical systems perspective without stochastic mechanics, but formalizing all other components and the formal robustness properties).

## Use Cases
- Any large-scale machine learning system that requires federated environments to train locally with as minimal aggregation updates (here, we use adaptive federated optimization to fit to heterogeneous data; adv. regularization adapts to perturbed data, and requires less samples (due to nsl))
- Deploying a federated system as the architecture for each machine learning model in a production pipeline with scalable optimization techniques already tested within image classifier models in the domain of image classification tasks.

## Future Work
- create secure gRPC connection between client nodes and trusted aggregator parent node
- run with ray as a distributed local graph specified by model and task, to aggregate federated environments
- add support for a set of additional certification algorithms of formal robustness as optimization
- dynamically analyze/extrapolate patterns from the set of robustness certification algorithms used for the federated environment of client-server models
- write protocol that can generalize across different neural network architectures with respect to federated environment
- add differential privacy as a precondition to evaluating formal/provable guarantees of the server/client models, and federated strategy (confidentiality --> privacy)
- use tfx and tensorflow.js in order to deploy a lightweight federated learning system that offloads latency expenses and computation to the server models with asynchronous updates during minimal traffic / load times if applicable. This way, companies can deploy scalable federated learning into their web applications, using centralized infrastructure with decentralized, scalable and robust learning. This applies to video and graphics web software. 

## Todos
- execute exp configs to get plots
- clean up codebase so that it can be used as a library
- readme: make diagram: Partition Dataset --> MNIST Image Sample --> NSL (Structured Signals, Feature Decomposition, Adv. Regularization) --> Core Deep Conv. Neural Network (nominally regularized is default setting) --> Federated Strategy --> Server-Side Parameter Evaluation (Trusted Aggregator) --> Certification : Check Against Adv. Robustness Property Specifications
