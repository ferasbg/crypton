#!/usr/bin/env python3
# Copyright (c) 2021 Feras Baig

import os
import pickle
import random
import sys

import keras
import tensorflow as tf

from verification.specification import RobustnessTrace 
from adversarial.main import Adversarial

"""

Note, we are initializing our constraint-satisfaction problem, our prepositional modal logic formula to assert robust output states after network state-transition T(x) for P(x,y) => Q(x,y) | x := y or input-output states match and are thus robust and thus correctness is proven. Negate non-robust output states and consider them in violation of the constraint specified.        

Note that this logic proves partial program correctness, since it's evaluating the input-output relations under the constraint of an adversarial attack on the network's inputs.

Convert convolutional network state into propositional formula given the constraint-satisfaction problem defined in trace.

Store some static variable state to indicate satisfiability in order to update the respective trace state to verify against specification given model checker.

Synthesize logical formula translated through encoding convolutional network as a constraint-satisfaction problem with respect to pre-condition and post-condition after network state-transition e.g. forwardpropagation. 

Note, the implication is satisfied given the network_postcondition e.g. output_state given perturbation norm and input_image as network_precondition

Note, according to Hoare logic with regards to propositional logic, the implementation of a function is partially correct with respect to its specification if, assuming the precondition is true just before the function executes, then if the function terminates, the postcondition is true

Note that x := input_class and y := output_class

Formally written as: Network ⊢ SAT ⟺ P(x,y) ⇒ Q(x,y) | P | ∀ x ∧ ∀ y     

Args:
    - network_precondition | Type := image_label_element isinstance tf.float32)
    - network_postcondition | Type := output_class isinstance int (corresponding to label in image_label_set)

Returns: Propositional Logic Formula Given Relationship Between Variable States 

# why do we return this formula and why do we verify examples not created

# how do we "verify" robustness? By confirming accuracy through some optimization problem we formulate our model into? And then say oh look it is still functional?
"""


'''Verify robustness given brightness perturbation norm-bounded attack against each input_image passed to network. '''
'''Verify robustness given l-norm (l^2, l-infinity, l-1) bounded perturbation attack against each input_image passed to network. '''
'''Model is considered robust given that under the pre-condition of fast sign gradient method attack, the model maintains the post-condition of correctness.'''