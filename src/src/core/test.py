import tensorflow_datasets as tfds
import numpy as np

import tensorflow_datasets as tfds
import numpy as np

# if as_supervised=False, it returns tuple, and if it's true, it returns a dict
dataset = tfds.load('mnist', as_supervised=True)
train = dataset['train']
test = dataset['test']
dataset = tfds.as_numpy(dataset)
