import os
import sys
import keras
import tensorflow as tf

from crypto.mpc_net import MPCNetwork


class BoundedMPCNetwork(MPCNetwork):
    raise NotImplementedError


if __name__ == '__main__':
    # create object instance of bounded network to be accessed by server node for bound propagation to compute on finite-state abstraction given MPCNetwork
    BoundedMPCNetwork()