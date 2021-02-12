import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import random
import pickle
import crypten


class MPCNetwork():
    """
        Deep Convolutional Neural Network With Secure Training and Testing

        Args:
            -
            -
            -

        Raises:


        Returns:


        References:
            - https://crypten.readthedocs.io/en/latest/nn.html
            - https://github.com/facebookresearch/CrypTen/blob/master/examples/mpc_autograd_cnn/mpc_autograd_cnn.py

        Examples:
            -
            -
            -

    """

    def __init__(self):
        super(MPCNet, self).__init__()
        # setup vgg given crypten.nn.Module network object, define network layers
        self.share0 = random.randrange(.001)
        self.share1 = random.randrange(.001)
        self.secret = {}
        self.party_1 = {}
        self.party_2 = {}


    def main():
        """Compute 3-Party MPC Training for MPCNetwork. Append nominal evaluation."""
        raise NotImplementedError

    def reconstruct(self, share0, share1):
        """Reconstruct computations, probably iterate as a subset set iterating over the kernels of the input matrix."""
        return (share0 + share1) % .0001

    def encrypt():
        raise NotImplementedError

    def decrypt():
        raise NotImplementedError



if __name__ = '__main__':
    MPCNet()
    MPCNet().main()

