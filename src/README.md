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
- Baseline 1: Data Corruption Regularization
- Baseline 2: Noise Corruption Regularization
- Baseline 3: Blur Corruption Regularization
- Control: Nominal Regularization

```python3
    blur_corruptions = ["motion_blur", "glass_blur", "zoom_blur", "gaussian_blur", "defocus_blur"]
    data_corruption_set = ["jpeg_compression", "elastic_transform", "pixelate"]
    noise_corruption_set = ["shot_noise", "impulse_noise", "speckle_noise"]
```

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

## Usage
- `python3 server.py --num_rounds=10 --strategy="fedadagrad"` 
- `python3 client.py --adv_grad_norm="infinity" --adv_multiplier=0.2 --adv_step_size=0.05 --batch_size=32 --epochs=1 --num_clients=10 nsl_reg=True`

## Remaining Features
- working: metrics fit to exp configs; write the code to write the data into the config objects that will create the plots
- working: writing funcs for server-side formal robustness metrics in `certify.Specification`, then add to `simulation.py` and `server.py` and log the certification metadata when calling the functions. Formalization may involve raw ε-robust accuracy, but either way is concrete even given 2-3 possible additional steps of abstraction. 
- test simulation.py in isolation in terms of default args to test if the processes and code all work
- write client-level, server-level, exp_config-level data to logfiles that can be parsed and used for creating plots.
- add feature of plot creation and writing the plot data to be used to create the end plots; this a question on how much is going in the file because that affects what data can be used to create the plot (perhaps the iterations have to take place in the same file instance if you want to get each coordinate pair)
- separate by plot specifications
