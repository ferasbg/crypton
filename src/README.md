# Crypton Core
Execute experimental configurations to assess the most optimal configuration for a production-level federated system that is robust against adversarial attacks fixed by norm type and norm values. Though it's not fixed to norm type nor norm value. Attack vectors aren't limited to adversarial examples (adv. input generation).


## Notes
- create virality around a research project out of boredom; interoperability and clear goals regarding research can help make source project act as catalyst if it's useful and well-written code, and customizable and adaptable to people's needs
- diagrams to add involve the backend implementation using nsl and flwr
- diagrams involving nsl and dcnn and federated adagrad
- diagrams for system architecture
- it's possible to have a paper + code configurable project and then have others contribute to the collective project that builds on the layers of verification and other neural network types and adv. regularization techniques etc. --> robustness networks for federated environments

## Usage
Run `exp-config-run.sh` to run all the experimental configurations to assess.

## Features
- done: encode partitions for 10 clients
- done: apply image corruptions to data along with perturb_batch processed in nsl backend before processed with flwr.client
- done: parse_args for exp config
- done: fed. adv. metrics
- formal robustness metrics
- "single-machine" simulation to aggregate client-server process

## Bugs


## Notes
- We can measure MNIST + adv_grad_norm + robust federated server-side accuracy based on the ε value.
