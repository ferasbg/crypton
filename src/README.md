# `crypton.src`
Crypton's core backend components and engine.

## Class Components
- `crypton.src.Prediction`: store base class for DCNN
- `crypton.src.Analytics`: store stats algorithms to compute metrics
- `crypton.src.Monitor`: collect metrics at runtime for all components, render statistical significance with analytics class
- `crypton.src.Verification`: store class for formal verification for DCNN and signal temporal logic
- `crypton.src.Adversarial`: adversarial nodes for DCNN
- `crypton.src.Client`: client to render analytics and processes
- `crypton.src.Deploy`: setup instance to train and test network
- `crypton.src.Crypto`: store algorithms for cryptographic scheme

