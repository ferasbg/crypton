# Adversarially Robust Federated Optimization with Neural Structured Learning
The purpose of this work is to build an adversarially robust federated system by utilizing adaptive and non-adaptive federated optimization techniques, as well as optimizing the adversarial regularization techniques used at the client-level in order to build a robust trusted aggregator model. More specifically, we aim to use neural structured learning and adaptive federated optimization together to build an adversarially robust federated learning system that can adapt to heterogeneous, sparse, and perturbed data that would be faced in a production-level federated environment.

## Introduction
- State the problem being solved and the approach taken.
- State what your goals for the paper are.
- "This paper introduces a novel and scalable connection between adversarial regularization and neural structured learning for production-level federated learning."

## Problem Statement
- Formalize the problem that is being solved in terms of the components responsible, neglecting unnecessarily specific variables that are default settings and thus preconditional assumptions (safe to assume, so extraneous in problem formulation).

## Crypton Architecture
- Partition Dataset --> MNIST Image Sample --> NSL (Structured Signals, Feature Decomposition, Adv. Regularization) --> Core Deep Conv. Neural Network (nominally regularized is default setting) --> Federated Strategy --> Server-Side Parameter Evaluation (Trusted Aggregator) --> Certification : Check Against Adv. Robustness Property Specifications

## Results and Evaluation
- Store results given each experiment configuration defined in `exp-config-run.sh` that will iteratively process train/test client partitions and client/model/server/strategy configurations to check what optimizations conform best to the specifications for the certification component.

## Research Innovations
- Implementing adversarial robustness certification for adversarially regularized deep convolutional networks in a federated setting, and using adaptive federated optimization to adapt to the real-world environment of data sparsity, corruptions, and heterogeneity. 
- First implementation that formalizes neural structured learning algorithm as an adversarial regularization technique combined with other forms of regularization under federated setting with respect to adversarial robustness certifications for a server-side trusted aggregator model.
- state the utility of why your project even matters in the grand scheme of certifiably robust federated CV models in production systems, and defining confidentiality given that you do not have differential privacy in the scope of your project, since confidentiality and certification of adversarial robustness in a federated ML system was the goal in the first place.
## Discussion
- Statements must either disprove, prove, certify, or validate the behavior, properties, and entropy (as implicit attitude in discussion without mathematical formalization of entropy itself other than the entropy method used) involved with the results that accounts for the default settings and experiment configs for each component.
- Reason about why particular behavior was observed.
- State the importance, validity, extensibility, reliability, and significance of the findings observed and reasoned about in the Discussion section.
- State the research innovations made on the paper, and how the information itself can be used (dynamical systems perspective without stochastic mechanics, but formalizing all other components and the formal robustness properties).

## System Components
- In `client.py`, the component at the client-level is split into the following subcomponents: `AdvRegClient` which is the client that uses the model of type `nsl.AdversarialRegularization`, `AdvRegClientConfig` which stores configuration data for the `AdvRegClient`, `Client` which uses the base `tf.keras.models.Model` object for its model, `ClientConfig`, `ExperimentConfig` which stores the arguments passed when the user runs `python3 client.py` after instantiating an insecure gRPC instance with `python3 server.py`, and `DatasetConfig`, which creates the client partitions depending on the argument for the total number of clients, and passes the loaded client train & test partitions to the `ClientConfig` or `AdvRegClientConfig` object respectively. By default, the `HParams` utilities object is also used, which defines the adversarial and base network hyperparameters for the models used by the clients.
- In `server.py`, the component at the server-level is split into the `ServerConfig` and `StrategyConfig` objects that configure the strategy and server-side trusted aggregator model that doesn't use adversarial regularization since it's the model used for server-side parameter evaluation.

## Future Work
- create secure gRPC connection between client nodes and trusted aggregator parent node
- run with ray as a distributed local graph specified by model and task, to aggregate federated environments
- add support for a set of additional certification algorithms of formal robustness as optimization
- dynamically analyze/extrapolate patterns from the set of robustness certification algorithms used for the federated environment of client-server models
- write protocol that can generalize across different neural network architectures with respect to federated environment
- add differential privacy as a precondition to evaluating formal/provable guarantees of the server/client models, and federated strategy (confidentiality --> privacy)


