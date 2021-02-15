import os
import pickle
import random
import sys

import keras
import matplotlib as plt
import numpy as np
import tensorflow as tf
from prediction.network import Network


class ReachabilityAnalysis():

    '''
    Compute Reachability Analysis with Symbolic Intervals on Deep Convolutional Neural Network

    Args (Properties):
        - self.reach_point: defined timestep and object to store network state given initialized metadata and properties of neural network
        - self.reach_option: analyzing different subsets of object state at future timesteps
        - self.reachSet: reachable set before pixel classification layer
        - self.ground_truth: store matrix with tuples (ex: ['image[0][0]', 'road']) that store label for each pixel of 224x224 image
        - self.reach_time: store computation time to compare against other verification tasks


    Returns:
        Type: ReachabilityAnalysis

    Raises:
        Error if Any Variable Element in Set of Access Points is NULL

    References:
        - https://arxiv.org/abs/1805.02242

    '''
    def __init__(self):
        self.reach_point = []
        self.reach_option = []
        self.reachSet = []
        self.ground_truth = []
        self.reach_time = 0

    def parse(self):
        
        '''
            Description: Parser to retrieve all required variable nodes from network at various timesteps

            Args:

            Returns:
            
            Raises:
            
            References:
            
            Examples:    
        
        '''
        # Network.getVariable()
        raise NotImplementedError


    raise NotImplementedError


if __name__ == '__main__':
    ReachabilityAnalysis()
    # compute given access points to network during training and testing
