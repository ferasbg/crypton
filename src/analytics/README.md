# `src.analytics`
Store analytics and statistical metrics for `crypton.prediction`, `crypton.verification`, `crypton.adversarial`, `crypton.crypto`. Store metadata for every component involved (ex: tracking algorithm output & performance, tracking instance metadata, tracking formal specifications & violations). Some operations will be asynchronous and synchronous. Important to understand statistical significance to indicate performance, system reliability/dependability. During runtime, `src.analytics` will compute statistical significance given data processed from `src.monitor`.


## Class Components
- `src.analytics.main`: compute statistical metrics
- `src.analytics.analytics_utils`: store logic to compute statistical metrics, whereas `src.analytics.main` will act as router to client



## Hardware Metadata
- network latency
- floatingpoint operations per second (FLOPS)
- memory management
- callbacks and workflows
- time and space complexity


## Other Notes
- Indicate a signature-based proof (some zero-trust authentication method) to initiate initial computation and proceeding computations. `crypton.adversarial` and `crypton.crypto` must both function together to prove confidentiality and integrity given model interpretability, defense against adversarial perturbations, and system faults (byzantine). The formal specifications will be meant to prove that the schemes were dependable and consistent.
- Note that input stream confidentiality for input layer of network along with oblivious activation functions and oblivious matrices that store weights of synapses between nodes (neurons) and the input neurons, can be measured through K-anonymity.

