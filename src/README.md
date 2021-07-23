# Crypton: Adversarially Robust Federated Optimization with Neural Structured Learning
The purpose of this work is to build an adversarially robust federated system by utilizing adaptive and non-adaptive federated optimization techniques, as well as optimizing the adversarial regularization techniques used at the client-level in order to build a robust trusted aggregator model. More specifically, we aim to use neural structured learning and adaptive federated optimization together to build an adversarially robust federated learning system that can adapt to heterogeneous, sparse, and perturbed data that would be faced in a production-level federated environment.

## Objective
- Execute experimental configurations to assess the most optimal configuration for a production-level federated system that is robust against adversarial attacks fixed by norm type and norm values. Though it's not fixed to norm type nor norm value. Attack vectors aren't limited to adversarial examples (adv. input generation).

## Technical Overview
- In `server.py`, the component at the server-level is the model used for server-side parameter evaluation. This is executed after the client-level process is computed in terms of its args-based configs in `simulation.py`.
- In `client.py`, the component at the client-level is split into the following subcomponents: `AdvRegClient` which is the client that uses the model of type `nsl.AdversarialRegularization`. `Client` which uses the base `tf.keras.models.Model` object for its model. 
- `DatasetConfig`, which creates the client partitions depending on the argument for the total number of clients, and passes the loaded client train & test partitions to the `Client` or `AdvRegClient` object respectively. 
- By default, the `HParams` utilities object is also used, which defines the adversarial and base network hyperparameters for the models used by the `AdvRegClient` and `Client` objects.
- In `simulation.py`, the system is formalized in terms of the server and client execution process defined with `multiprocessing.Process`. The experimental configuration defined by `create_args_parser` configures the `server_args : ArgumentParser` and `client_args : ArgumentParser` for their respective processes. 
- When executing the experimental configurations, we iterate over every combination and sequentially utilize `client.DatasetConfig` to handle data processing, `simulation.PlotsConfig` to ingest and configure the metrics for the current experimental config and creating the plots with `utils.Plot` given the `simulation.PlotConfig` object. 
- In `certify.py`, we utilize the `certify.Specification` class which defines all the adversarial robustness properties to target the subcomponents of the federated system, eg the server-side trusted aggregator model in `server.py`. The objective of the properties and their checking methods are to certify, assert, and measure the formalized adversarial robustness of the federated system and testing what system-level (adaptive/non-adaptive federated strategy, adversarial regularization technique(s)) configurations map best to its real-world production-level stability, dependability, and reliability.

## Context
- It's important to utilize adversarial regularization whether or not the data is non-IID or IID for machine learning models in production systems.

## Adversarial Regularization Techniques
- Target Technique: Neural Structured Learning
- Baseline 1: Data Corruption-Regularized Learning
- Baseline 2: Noise Corruption-Regularized Learning
- Baseline 3: Blur Corruption-Regularized Learning
- Control: Nominal Regularization

## Figures
 Method | Communication Cost     | Server-Side Model Accuracy-Under-Attack | Server-Side Model Adversarial Loss
| --- | ---| ---|---|
| FedAvg + NSL           | Null | X%                                |  Y%
| FedAvg + GaussianNoise | Null | X%                                |  Y%
| FedAvg + GaussianNoise | Null | X%                                |  Y%
| FedAvg + GaussianNoise | Null | X%                                |  Y%
| FedAvg + GaussianNoise | Null | X%                                |  Y%
| FedAvg + GaussianNoise | Null | X%                                |  Y%
| FedAvg + GaussianNoise | Null | X%                                |  Y%
| FedAvg + GaussianNoise | Null | X%                                |  Y%
| FedAvg + GaussianNoise | Null | X%                                |  Y%
| FedAvg + GaussianNoise | Null | X%                                |  Y%
| FedAvg + GaussianNoise | Null | X%                                |  Y%
| FedAvg + GaussianNoise | Null | X%                                |  Y%

## Todos
- resolve server-client process when running on `.sh` thread
- write data to a logfile continuously in terms of its target plot, and after the .sh file is done running, make plots given written data. Do this for each exp config set (each `.sh` file there is --> 4) 
- get model robustness metrics (formal, nominal) --> nominal to keep the scope, formal for the other paper (you can make 2 papers out of this?)
- make tables, diagrams, graphs/plots for the paper in the `paper` directory in `/docs/`. Write arxiv paper that clearly bolsters research innovations and results.
- resolve FedAdagrad and FaultTolerantFedAvg
- get the logfile --> plot feature working at the `trials.sh` level before writing iterative scripts in `dev/fedavg`. Then support FedAdagrad, and then support getting nomina and formal robustness.

