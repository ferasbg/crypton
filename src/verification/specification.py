import os
import sys


import torch
import torchvision


from prediction.network import Network
from hyperproperties import HyperProperties, RobustnessProperties, SafetyProperties, LivenessProperties


class Specification():
    """Core Formal Specifications for Deep Convolutional Neural Network. Write and aggregate all specifications and sub-nodes in verification node to compute on network during its training and testing

    Args:
        self.ibp = IntervalBoundPropagation(): store algorithm for input bound propagation
        self._robustness_properties = RobustnessProperties(): store initialized robustness properties and compute specifications for robustness verification
        self._safety_properties = SafetyProperties(): : store initialized safety properties and compute specifications for safety verification
        self._liveness_properties = LivenessProperties(): : store initialized liveness properties and compute specifications for liveness verification

    Returns:
       Type: Specification ( object that stores metrics for all computations for safety verification, robustness, and liveness properties)

    """

    def __init__(self):
        # store hyperproperty objects and functions to compute verification algorithms
        self.ibp = IntervalBoundPropagation()
        self._robustness_properties = RobustnessProperties()
        self._safety_properties = SafetyProperties()
        self._liveness_properties = LivenessProperties()

    def main(self):
        pass


if __name__ == '__main__':
    Specification()

