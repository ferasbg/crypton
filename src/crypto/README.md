# `crypton.src.crypto`
store logic for mpc security properties of the neural network. 

## Class Components
- `crypton.src.crypto.mpc`: store MPC scheme and functions to apply to base neural network class with updated weights/architecture given training
- `crypton.src.crypto.ot`: nodes for oblivious transfer, parties to compute function in MPC scheme, primarily act as data structures for party nodes that compute secret shares of neural net during training on server node
- `crypton.src.crypto.mpc_net`: child class for base network, to re-use component in `src.deploy.secure_nn`, act as instance to run model under crypto constraints
- `crypton.src.crypto_stl`: compute STL on DCNN with MPC scheme, to compute object instance in `src.deploy`


## Research
- The parties first secret-share their inputs; i.e. input xi is shared so that ∑jxij=xi and party Pj holds xij (and Pi which provides input is included in this sharing, even though it knows the sum).
- The parties perform additions and multiplications on these secret values by local computations and communication of certain values (in methods specified below). By construction, the result of performing an operation is automatically shared amongst the parties (i.e. with no further communication or computation).
- Finally, the parties 'open' the result of the circuit evaluation. This last step involves each party sending their 'final' share to every other party (and also performing a check that no errors were introduced by the adversary along the way).

