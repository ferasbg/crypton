# crypton.src.crypto
 

## Class Components
- `crypton.src.crypto.mpc`: store MPC scheme and functions to apply to base neural network class with updated weights/architecture given training
- `crypton.src.crypto.ot`: nodes for oblivious transfer, parties to compute function in MPC scheme, primarily act as data structures for party nodes that compute secret shares of neural net during training on server node
- `crypton.src.crypto.mpc_net`: child class for base network, to re-use component in `src.deploy.secure_nn`
- `crypton.src.crypto_stl`: compute STL on DCNN with MPC scheme, to compute object instance in `src.deploy`

## Other Notes
This is the current general schema but make sure to update it with respect to CrypTen framework and Pysyft.


