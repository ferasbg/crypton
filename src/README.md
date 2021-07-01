# Crypton Core
Execute experimental configurations to assess the most optimal configuration for a production-level federated system that is robust against adversarial attacks fixed by norm type and norm values. Though it's not fixed to norm type nor norm value. Attack vectors aren't limited to adversarial examples (adv. input generation).

## Usage
- `python3 server.py --num_rounds=10 --strategy="fedadagrad"` 
- `python3 client.py --num_partitions=10 --adv_grad_norm="infinity" --adv_multiplier=0.2 --adv_step_size=0.05 --batch_size=32 --epochs=5 --num_clients=10 adv_reg=True gaussian_layer=True`

## Features
- working: server-side evaluation after fit_round and sample_round sampling of clients
- working: writing funcs for formal robustness metrics
- working: write simulation.py that takes in args per execution instance and runs client-server processes similar to flwr example for simulation.py
- certification of adv. robustness: compute formal robustness metrics based on the formalization paper
- exp-config-run.sh is an iterative for loop that starts a server, and runs the clients based on the partition specified by index and samples all the clients with the server/client/exp config as args to each target file and then shuts down the server and then starts a new one to run the next test, until all the configs are run (all combinations --> defined with itertools)

## Research Notes
- fedadagrad adaptability + feature decomposition from NSL / higher dimensionality of features + DCNN with skip connections and nominal regularizations etc --> converge to satisfy robustness specifications and conform to optimal optimization formulation
- structured signals (higher dimensionality of feature representation) + adaptive federated optimization --> more robust model with corrupted and sparse data;
- strategy (when explaining/comparing GaussianNoise and without GaussianNoise): Other implementations apply the perturbation epsilon deeper into the network, but for maintaining dimensionality (and other reasons specified in the paper), the earlier the perturbations applied, the better (Goodfellow, et. al --> adversarial examples paper).
- adversarial neighbors and NSL core protocol relation to fedAdagrad
