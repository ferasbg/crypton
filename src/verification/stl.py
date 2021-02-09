# Copyright 2021 Feras Baig

'''

Given defined trace properties in `verification.hyperproperties`, compute and iterate over each trace property and track if the computed (symbolic) state abstraction produces a counter-example of the property trace.


'''
import os
import time
import random
import torch
from torch import nn

from prediction.network import Network
from hyperproperties import HyperProperties, RobustnessProperties, SafetyProperties, LivenessProperties
from bound_propagation import BoundPropagation



class STLSpecification(Network):
    """Compute property inferencing and checking given formal specifications in `hyperproperties`, which stores all of the trace properties, and `verification.stl` will process the functions and variables of the hyperproperties, which will have computed the data that accesses the network state to model the network state to compute all the verifications, for which reachability sets will be computed in `symbolic_representation.py` and the network state and the computational model for the network state for the specifications to be computed will be stored in hyperproperties. The `bound_propagation` file is meant to compute the upper and lower bounds for the reachability problem, adjacent to the other processes.

    Args:
        - self.robustness = RobustnessProperties()
        - self.safety = SafetyProperties()
        - self.liveness = LivenessProperties()
    """

    def __init__(self):
        self.robustness = RobustnessProperties()
        self.safety = SafetyProperties()
        self.liveness = LivenessProperties()



class Trace():

    def __init__(self):
        self.l2norm = 0
        self.upperBound = 0
        self.lowerBound = 0
        self.reachableSets = []

    @staticmethod
    def check_trace(self):
        raise NotImplementedError

    pass

class CheckTraceData():
    pass



if __name__ == '__main__':
    Trace.check_trace()

