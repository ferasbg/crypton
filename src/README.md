# Crypton: Adversarially Robust Federated Optimization with Neural Structured Learning
The purpose of this work is to build an adversarially robust federated system by utilizing adaptive and non-adaptive federated optimization techniques, as well as optimizing the adversarial regularization techniques used at the client-level in order to build a robust trusted aggregator model. More specifically, we aim to use neural structured learning and adaptive federated optimization together to build an adversarially robust federated learning system that can adapt to heterogeneous, sparse, and perturbed data that would be faced in a production-level federated environment.

## Objective
Execute experimental configurations to assess the most optimal configuration for a production-level federated system that is robust against adversarial attacks fixed by norm type and norm values. Though it's not fixed to norm type nor norm value. Attack vectors aren't limited to adversarial examples (adv. input generation).

## Technical Overview
- In `server.py`, the component at the server-level is the model used for server-side parameter evaluation. This is executed after the client-level process is computed in terms of its args-based configs in `simulation.py`.
- In `client.py`, the component at the client-level is split into the following subcomponents: `AdvRegClient` which is the client that uses the model of type `nsl.AdversarialRegularization`. `Client` which uses the base `tf.keras.models.Model` object for its model. 
- `DatasetConfig`, which creates the client partitions depending on the argument for the total number of clients, and passes the loaded client train & test partitions to the `Client` or `AdvRegClient` object respectively. 
- By default, the `HParams` utilities object is also used, which defines the adversarial and base network hyperparameters for the models used by the `AdvRegClient` and `Client` objects.
- In `simulation.py`, the system is formalized in terms of its components and subcomponents compartmentalized by the execution process defined with `multiprocessing` since each `ExperimentalConfig` object is the instantiation of the server and client processes that are configured with `ArgumentParser`. We iterate over every combination and sequentially utilize `utils.Plot` and `client.DatasetConfig` to handle data processing and metrics. 
- In `certify.py`, we utilize the `certify.Specification` class which defines all the adversarial robustness properties to target the subcomponents of the federated system, eg the server-side trusted aggregator model in `server.py`. The objective of the properties and their checking methods are to certify, assert, and measure the formalized adversarial robustness of the federated system and testing what system-level (adaptive/non-adaptive federated strategy, adversarial regularization technique(s)) configurations map best to its real-world production-level stability, dependability, and reliability.

## Adversarial Regularization Techniques
- Target Technique: Neural Structured Learning
- Baseline 1: Data Corruption Regularization
- Baseline 2: Noise Corruption Regularization
- Baseline 3: Blur Corruption Regularization

```python3
    blur_corruptions = ["motion_blur", "glass_blur", "zoom_blur", "gaussian_blur", "defocus_blur"]
    data_corruption_set = ["jpeg_compression", "elastic_transform", "pixelate"]
    noise_corruption_set = ["shot_noise", "impulse_noise", "speckle_noise"]
```

## Usage
- `python3 server.py --num_rounds=10 --strategy="fedadagrad"` 
- `python3 client.py --adv_grad_norm="infinity" --adv_multiplier=0.2 --adv_step_size=0.05 --batch_size=32 --epochs=1 --num_clients=10 nsl_reg=True`

## Remaining Features
- working: metrics fit to exp configs; write the code to write the data into the config objects that will create the plots
- working: writing funcs for server-side formal robustness metrics in `certify.Specification`, then add to `simulation.py` and `server.py` and log the certification metadata when calling the functions. Formalization may involve raw ε-robust accuracy, but either way is concrete even given 2-3 possible additional steps of abstraction. 

## Bugs
- errors with iterative exp config execution body
- errors with graphing iterative exp config and defining data for plots during server-client process (formalize analysis via plots/data)
