# Crypton Core
Execute experimental configurations to assess the most optimal configuration for a production-level federated system that is robust against adversarial attacks fixed by norm type and norm values. Though it's not fixed to norm type nor norm value. Attack vectors aren't limited to adversarial examples (adv. input generation).

## Usage
Run `exp-config-run.sh` to run all the experimental configurations to assess.

## Features
- working: server-side evaluation after fit_round and sample_round sampling of clients
- working: writing funcs for formal robustness metrics
- working: "single-machine" simulation to aggregate client-server process

## Bugs
- steps_per_epoch doesn't apply to the cache that pulls 5 epochs and 1875 per epoch in train. How can I override this?

## TODO
- exp: We can measure MNIST + adv_grad_norm + robust federated server-side accuracy based on the ε value.
- execute each permutation/combination for all possible exp configs, record metrics and state during train/eval
- certification of adv. robustness: compute formal robustness metrics based on the formalization paper
- create virality around a research project out of boredom; interoperability and clear goals regarding research can help make source project act as catalyst if it's useful and well-written code, and customizable and adaptable to people's needs
- diagrams to add involve the backend implementation using nsl and flwr
- diagrams involving nsl and dcnn and federated adagrad
- diagrams for system architecture (NSL Graph Architecture --> Core DCNN --> Federated Environment --> Specification (Certification of Robustness))
- it's possible to have a paper + code configurable project and then have others contribute to the collective project that builds on the layers of verification and other neural network types and adv. regularization techniques etc. --> robustness networks for federated environments
- get rid of all files irrelevant to project, scrap code, test code, etc
- make formal diagrams akin to patent art and general research art (diagrams)
- add all mathematical formulations relevant to paper in its own document to extrapolate from
- write a document that formalizes the value prop from the paper itself
- define "future work" in README such that contributors can build on top of existing paper, akin to other seminal papers (certification of federated adv-regularized models, optimization and control)
- write formal unit tests 
- write the README that works with the arxiv preprint
- write tex file containing abstract and paper
- during arxiv preprint process: advertise/market the paper on HN, reddit, twitter, discord, etc as a benchmark for further research in certification of robust NNs, optimization, etc

# Notes
- fedadagrad adaptability + feature decomposition from NSL / higher dimensionality of features + DCNN with skip connections and nominal regularizations etc --> converge to satisfy robustness specifications and conform to optimal optimization formulation

## Remaining Research Questions
- how does convexity apply to the optimizer used to most efficiently aggregate the learnings of each client on local data? Surely important considering optimization formulation is interlinked with specifications that depend on measuring variability.

## Code Comments
        - there are different regularization techniques, but keep technique constant
        - formalize relationship between adversarial input generation for a client and its server-side evaluation-under-attack.
        - adversarial regularization is very useful for defining an explicit structure e.g. structural signals rather than single samples.
        - nsl-ar structured signals provides more fine-grained information not available in feature inputs.
        - We can assume training with robust adversarial examples makes it robust against adversarial perturbations during inference (eval), but how does this process fair when there's a privacy-specific step of using client models to locally train on this data and use a federated optimization technique for server-side evaluation? How can we utilize unsupervised/semi-supervised learning and these "structured signals" to learn hidden representations in perturbed or otherwise corrupted data (image transformations) with applied gaussian noise (these configurations exist to simulate a real-world scenario). We want to then formalize this phenomenon and the results between each experimental configuration.
        - adv reg. --> how does this affect fed optimizer (regularized against adversarial attacks) and how would differences in fed optimizer affect adv. reg model? Seems like FedAdagrad is better on het. data, so if it was regularized anyway with adv. perturbation attacks, it should perform well against any uniform of non-uniform or non-bounded real-world or fixed norm perturbations.
        - wrap the adversarial regularization model to train under two other conditions relating to GaussianNoise and specified perturbation attacks during training specifically.
        - graph the feature representation given graph with respect to the graph of the rest of its computations, and the trusted aggregator eval
   
