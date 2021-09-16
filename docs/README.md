# Adversarially Robust Federated Optimization with Neural Structured Learning
The purpose of this work is to build an adversarially robust federated system by utilizing adaptive and non-adaptive federated optimization techniques, as well as optimizing the adversarial regularization techniques used at the client-level in order to build a robust trusted aggregator model. More specifically, we aim to use neural structured learning and adaptive federated optimization together to build an adversarially robust federated learning system that can adapt to heterogeneous, sparse, and perturbed data that would be faced in a production-level federated environment.

## Introduction
- There is a disconnect in terms of system design relating to federated systems for specifically image recognition and other computer vision tasks when it comes to performing well on real-world data. Models converge well on fixed, IID data with unrealistic quality, sparsity, fault tolerance, adaptivity, and so on, so this system is built to adapt to the data state and the system state (adaptive strategies, adversarially regularized client models). The goal for this paper is to integrate scalable optimization techniques both at the model level and at the system level that can utilize adaptivity and robustness against real-world data that has surface variations, perturbations, corruptions, and other natural distortions, not to mention its non-IID nature and sparsity. Our system can handle all of these environmental variables.
- This paper introduces a novel and scalable connection between adversarial regularization and neural structured learning for production-level federated learning.

## Problem Statement
Implement a scalable connection between adaptive server-side federated optimization and client-side adversarial regularization in order to build a adversarially robust federated machine learning system.

## Crypton Architecture

<div align="center">
    <img src="https://github.com/ferasbg/crypton/blob/master/docs/media/nsl_architecture.png" width="800" align="center">
</div>

#### System Components
- `client.py`: define the client models and their configuration subcomponents.
- `server.py`: define the server-side trusted aggregator model with adversarial perturbations applied to the evaluation data, as well as defining the adaptive/non-adaptive federated strategy.   
- `simulation.py`: aggregate server-client processes iteratively in terms of the experimental configuration permutations
- `HParams`: adversarial hyperparameter settings
- `AdvRegClient`: client that uses neural structured learning
- `Client`: client that uses nominal regularization techniques and can be configured for corruption learning

## Research Innovations
- Implementing adversarially regularized deep convolutional networks in a federated setting, and using adaptive federated optimization. The purpose of which are to build a system that can adapt to the real-world environment of data sparsity, corruptions, and heterogeneity. 
- First implementation that formalizes neural structured learning algorithm as an adversarial regularization technique combined with other forms of regularization under federated setting with respect to adversarial robustness certifications for a server-side trusted aggregator model.
- Companies can use the base architecture into their own machine learning systems in terms of their individual components that involve singularly defined tasks in their end-to-end system, and define federated (cluster) environments for each of their components while maintaining scalability and robustness.

## Discussion
- State the importance, validity, extensibility, reliability, and significance of the findings observed.

## Use Cases
- Any large-scale machine learning system that requires federated environments to train locally with as minimal aggregation updates (here, we use adaptive federated optimization to fit to heterogeneous data; adv. regularization adapts to perturbed data, and requires less samples (due to nsl))
- Deploying a federated system as the architecture for each machine learning model in a production pipeline with scalable optimization techniques already tested within image classifier models in the domain of image classification tasks.

## Future Work
- add support for a set of additional certification algorithms of formal robustness as optimization
