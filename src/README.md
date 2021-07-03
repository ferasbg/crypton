# Crypton Core
Execute experimental configurations to assess the most optimal configuration for a production-level federated system that is robust against adversarial attacks fixed by norm type and norm values. Though it's not fixed to norm type nor norm value. Attack vectors aren't limited to adversarial examples (adv. input generation).

## Usage
- `python3 server.py --num_rounds=10 --strategy="fedadagrad"` 
- `python3 client.py --num_partitions=10 --adv_grad_norm="infinity" --adv_multiplier=0.2 --adv_step_size=0.05 --batch_size=32 --epochs=5 --num_clients=10 adv_reg=True gaussian_layer=True`

## Features
- working: server-side evaluation after fit_round and sample_round sampling of clients
- working: writing funcs for formal robustness metrics
- working: write simulation.py that takes in args per execution instance and runs client-server processes similar to flwr example for simulation.py
- certification of adv. robustness (formal statements are written and functions can be used to compute satisfiability of adversarial robustness properties): compute formal robustness metrics based on the formalization paper
- exp-config-run.sh is an iterative for loop that starts a server, and runs the clients based on the partition specified by index and samples all the clients with the server/client/exp config as args to each target file and then shuts down the server and then starts a new one to run the next test, until all the configs are run (all combinations --> defined with itertools)

## Technical Notes
- metrics should get plots for what relationships you want to measure given exp config combination for all combinations

```python
for exp_config in exp_config_set:
    # create strategy object given strategy config that processes args.strategy 
    strategy = StrategyConfig(args.strategy).strategy
    start_server()
    for i in range(args.num_clients):
        adv_client_config = AdvRegClientConfig(model=[], params=[], train_dataset=[], test_dataset=[], validation_steps=[])
        client = AdvRegClient()
        flwr.server.start_server(server_address=DEFAULT_SERVER_ADDRESS, strategy=strategy)
        flwr.client.start_keras_client(server_address=DEFAULT_SERVER_ADDRESS, client=client)
        partition_train = load_train_partition(i)
        partition_test = load_test_partition(i)
        start_client(exp_config, client, partition_train, partition_test)
        
    server_model.evaluate(perturbation_attack_dataset)
    exp_config.log_experiment()


```
- The `exp-config-run.sh` file will be an iterative shell file that will run each experimental configuration permutation.

## Technical Overview
- In `client.py`, the component at the client-level is split into the following subcomponents: `AdvRegClient` which is the client that uses the model of type `nsl.AdversarialRegularization`, `AdvRegClientConfig` which stores configuration data for the `AdvRegClient`, `Client` which uses the base `tf.keras.models.Model` object for its model, `ClientConfig`, `ExperimentConfig` which stores the arguments passed when the user runs `python3 client.py` after instantiating an insecure gRPC instance with `python3 server.py`, and `DatasetConfig`, which creates the client partitions depending on the argument for the total number of clients, and passes the loaded client train & test partitions to the `ClientConfig` or `AdvRegClientConfig` object respectively. By default, the `HParams` utilities object is also used, which defines the adversarial and base network hyperparameters for the models used by the clients.
- In `server.py`, the component at the server-level is split into the `ServerConfig` and `StrategyConfig` objects that configure the strategy and server-side trusted aggregator model that doesn't use adversarial regularization since it's the model used for server-side parameter evaluation.

## Research Notes
- math is involved around decision formulations that use formal notation of functions used on client-level, the federated strategy, the nsl-specific math, and the formalization of different perturbations/configs as regularization etc
- fedadagrad adaptability + feature decomposition from NSL / higher dimensionality of features + DCNN with skip connections and nominal regularizations etc --> converge to satisfy robustness specifications and conform to optimal optimization formulation
- structured signals (higher dimensionality of feature representation) + adaptive federated optimization --> more robust model with corrupted and sparse data;
- strategy (when explaining/comparing GaussianNoise and without GaussianNoise): Other implementations apply the perturbation epsilon deeper into the network, but for maintaining dimensionality (and other reasons specified in the paper), the earlier the perturbations applied, the better (Goodfellow, et. al --> adversarial examples paper).
- adversarial neighbors and NSL core protocol relation to fedAdagrad
- how does convexity apply to the optimizer used to most efficiently aggregate the learnings of each client on local data? Surely important considering optimization formulation is interlinked with specifications that depend on measuring variability.

## Misc
- there are different regularization techniques, but keep technique constant
- formalize relationship between adversarial input generation for a client and its server-side evaluation-under-attack.
- adversarial regularization is very useful for defining an explicit structure e.g. structural signals rather than single samples.
- nsl-ar structured signals provides more fine-grained information not available in feature inputs.
- Isolate the regularization techniques (Gaussian, Corruptions, Neural Structured Learning) to simplify things. If it makes sense to combine particular regularizations, then go for it. I do think that all nominal regularization techniques should be shared but all the adversarial regularization techniques should be isolated per client-server trial.
- We can assume training with robust adversarial examples makes it robust against adversarial perturbations during inference (eval), but how does this process fair when there's a privacy-specific step of using client models to locally train on this data and use a federated optimization technique for server-side evaluation? How can we utilize unsupervised/semi-supervised learning and these "structured signals" to learn hidden representations in perturbed or otherwise corrupted data (image transformations) with applied gaussian noise (these configurations exist to simulate a real-world scenario). We want to then formalize this phenomenon and the results between each experimental configuration.
- adv reg. --> how does this affect fed optimizer (regularized against adversarial attacks) and how would differences in fed optimizer affect adv. reg model? Seems like FedAdagrad is better on het. data, so if it was regularized anyway with adv. perturbation attacks, it should perform well against any uniform of non-uniform or non-bounded real-world or fixed norm perturbations.

## Adversarial Regularization Techniques
The goal is to assess what optimization techniques (as regularization) help model convergence at the client-level to start answering the question with what strategy it works best with in order to help the server-side model against norm-bounded perturbation attacks during evaluation.

Theoretically, you can make relationships between the ideas of surface variations as non-uniform perturbations that help the models converge well without overfitting. But you can also argue that forms of "data augmentation" done through the technique formalized by neural structured learning (which generates non-convex norm-bounded sets based on the input sample as neighbors to regularize the model). You can then relate these ideas to the data and the strategy. For an example, if your aggregation strategy to update the server model with the least amount of data (sparsity) and computation (in a production system, this is crucial), then relating both the ideas of the most optimal adversarial regularization technique and what strategy it works best with based on how it configures client-level data can help build more robust federated learning systems in general.

- Target Technique: Neural Structured Learning
- Baseline Set C: Image Degradation
- Baseline Set N: Gaussian Noise Regularization

```python3
blur_corruptions = ["motion_blur", "glass_blur", "zoom_blur", "gaussian_blur", "defocus_blur"]
data_corruption_set = ["jpeg_compression", "elastic_transform", "pixelate"]
noise_corruption_set = ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"]

```


## ExperimentalConfig
- It's important to formalize both what experimental configs must be defined and how they map and translate into the graphs that will be plotted and tables that will be defined (when comparing against baseline, etc)
- It depends on what variables need to be isolated and held constant. The main variables in question for the client and server are the strategy, adv_grad_norm, adv_step_size, 
