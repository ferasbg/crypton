# test data.py functions and attacks.py functions

from attacks import *
from data import *
from adversarial_regularization import *

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
