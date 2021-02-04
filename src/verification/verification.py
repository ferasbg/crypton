import os
import sys
import pickle
import random
import keras
import tensorflow


from verification.specification import Specification
from verification.hyperproperties import SafetyProperties, RobustnessProperties, LivenessProperties



class Verification():
    """Check each trace property defined in the formal specifications in `verification.specification`. Use solver to process bounded network state abstraction to compute satisfiability for each trace property.

    """
    def __init__(self):
        """Store the formal specifications and each trace property. Also store the state for the parameters required for each verification technique."""
        self.specification = Specification()
        self.safety_properties = SafetyProperties()
        self.robustness_properties = RobustnessProperties()
        self.liveness_properties = LivenessProperties()


    raise NotImplementedError


if __name__ == '__main__':
    Verification()
    # sequential workflow of property checking where `src.specification` stores the object state for each trace property whereas verification does the property check

