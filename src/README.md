# Crypton: Adversarially Robust Federated Optimization with Neural Structured Learning
The purpose of this work is to build an adversarially robust federated system by utilizing adaptive and non-adaptive federated optimization techniques, as well as optimizing the adversarial regularization techniques used at the client-level in order to build a robust trusted aggregator model. More specifically, we aim to use neural structured learning and adaptive federated optimization together to build an adversarially robust federated learning system that can adapt to heterogeneous, sparse, and perturbed data that would be faced in a production-level federated environment.

## Objective
Execute experimental configurations to assess the most optimal configuration for a production-level federated system that is robust against adversarial attacks fixed by norm type and norm values. Though it's not fixed to norm type nor norm value. Attack vectors aren't limited to adversarial examples (adv. input generation).

## Technical Overview
- In `server.py`, the component at the server-level is the model used for server-side parameter evaluation. This is executed after the client-level process is computed in terms of its args-based configs in `simulation.py`.
- In `client.py`, the component at the client-level is split into the following subcomponents: `AdvRegClient` which is the client that uses the model of type `nsl.AdversarialRegularization`. `Client` which uses the base `tf.keras.models.Model` object for its model. 
- `DatasetConfig`, which creates the client partitions depending on the argument for the total number of clients, and passes the loaded client train & test partitions to the `Client` or `AdvRegClient` object respectively. 
- By default, the `HParams` utilities object is also used, which defines the adversarial and base network hyperparameters for the models used by the `AdvRegClient` and `Client` objects.
- In `simulation.py`, the system is formalized in terms of the server and client execution process defined with `multiprocessing.Process`. The experimental configuration defined by `create_args_parser` configures the `server_args : ArgumentParser` and `client_args : ArgumentParser` for their respective processes. 
- When executing the experimental configurations, we iterate over every combination and sequentially utilize `client.DatasetConfig` to handle data processing, `simulation.PlotsConfig` to ingest and configure the metrics for the current experimental config and creating the plots with `utils.Plot` given the `simulation.PlotConfig` object. 
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

## Tests
- test simulation.py in isolation in terms of default args to test if the processes and code all work
- add feature of plot creation and writing the plot data to be used to create the end plots; this a question on how much is going in the file because that affects what data can be used to create the plot (perhaps the iterations have to take place in the same file instance if you want to get each coordinate pair)


## Comments
- 3 strats, 13 techniques, 2 norm types, 10 norm values; hardcode args per exp config per python3 simulation.py ran in experimental_config.sh
    - hardcode all your args and execute them as their own exp configs rather than executing all the permutations (which would take forever)
    - plot process: 1) collect data for each plot and add it to those plot objects inside MetricsConfig, 2) build the plots and update the variables, 3) write the plots to /figures directory

- store metrics in PlotsConfig object, then pass the data stored for each respective plot into the experiment metadata object to then build the list
- this is based on 1) the plots, 2) the exp_config data that can be collected per simulation instance
- it makes sense to iterate in terms of the target variables you used in terms of the plot to create (this can be specified and iteratively updated in the .sh file, and not inside of main)
- execute each exp config with simulation.py using main(args) --> run_simulation(args)
- separate by plot specifications
- variables: adversarial regularization technique, federated strategy (adaptive, non-adaptive), adv_grad_norm, adv_step_size
