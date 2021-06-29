# Crypton Core
Execute experimental configurations to assess the most optimal configuration for a production-level federated system that is robust against adversarial attacks fixed by norm type and norm values. Though it's not fixed to norm type nor norm value. Attack vectors aren't limited to adversarial examples (adv. input generation).


## Notes
- create virality around a research project out of boredom; interoperability and clear goals regarding research can help make source project act as catalyst if it's useful and well-written code, and customizable and adaptable to people's needs
- diagrams to add involve the backend implementation using nsl and flwr
- diagrams involving nsl and dcnn and federated adagrad
- diagrams for system architecture
- it's possible to have a paper + code configurable project and then have others contribute to the collective project that builds on the layers of verification and other neural network types and adv. regularization techniques etc. --> robustness networks for federated environments


## Bugs
- too many to unpack for MapDataset: nsl/flwr conflict with processing mnist eval data
    - for nsl it processes MapDataset fine, but for flwr.client I believe there's a conflict with how evaluate processes the data
- it's possible that either the AdvRegClient class needs a few updates to fit to general standards, compliance with backend ops for flwr.client, and re-writing their backend possibly
- figure out what types work with nsl given that they work with flwr: what types work with flwr.client? --> using direct tuples for nsl model? Test this
- reasoning for this bug: the tuples are tensors because they store np.ndarrays for the features e.g. image,label pairs; each Tensor is evaluated with .evaluate(). It's why other Dataset types may not be explicitly defined to be iterable. It's why there's an unpacking error.
- there is no .fit working actually when there's 313 images processed in the nsl cache; whether that's a validation method, I am unsure what code is touching that eager execution to occur
- todo: only solution: get to MapDataset, then convert to iterable tuple of dicts with FeatureDict

## Features
- done: encode partitions for 10 clients
- done: apply image corruptions to data along with perturb_batch processed in nsl backend before processed with flwr.client
- done: parse_args for exp config
- fed. adv. metrics
- formal robustness metrics
- "single-machine" simulation to aggregate client-server process

