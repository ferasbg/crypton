# `crypton.src`
Crypton's core backend components and engine.

## Class Components
- `crypton.src.prediction`: store base class for DCNN
- `crypton.src.analytics`: store stats algorithms to compute metrics
- `crypton.src.monitor`: collect metrics at runtime for all components, render statistical significance with analytics class
- `crypton.src.verification`: store class for formal verification for DCNN and signal temporal logic
- `crypton.src.adversarial`: adversarial nodes for DCNN
- `crypton.src.client`: client to render analytics and processes
- `crypton.src.deploy`: setup instance to train and test network
- `crypton.src.crypto`: store algorithms for cryptographic scheme



## Requirements
- missing: adversarial protocol implementation and testing strategies; sequential workflow for src.deploy (instance / runner), and the specific symbolic and numeric representation of the safety and robustness properties for the neural network; implementing smpc for an extended neural network
- setup access points for every node and its respective arguments necessary for each computation



