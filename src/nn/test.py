import keras
import tensorflow as tf
import os, sys
import matplotlib as plt

from network import Network

## test all functions here
Network.get_cifar_data()
input_image = Network.x_train[0]
print(input_image)