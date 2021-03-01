import keras
import tensorflow as tf
import os, sys
import matplotlib as plt

from network import Network

## test all functions here
network = Network()
# get eval after nominal training
train = network.train()
print(train)