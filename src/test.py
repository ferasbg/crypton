import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

def normalize(features):
  features['image'] = tf.cast(
      features['image'], dtype=tf.float32) / 255.0
  return features

def convert_to_tuples(features):
  return features['image'], features['label']

def convert_to_dictionaries(image, label):
  return {'image': image, 'label': label}

# # standard dataset processed for base client
# datasets = tfds.load('mnist')
# train_dataset = datasets['train']
# test_dataset = datasets['test']
# train_dataset_for_base_model = train_dataset.map(normalize).shuffle(10000).batch(batch_size=32).map(convert_to_tuples)
# test_dataset_for_base_model = test_dataset.map(normalize).batch(batch_size=32).map(convert_to_tuples)
# train_dataset_for_adv_model = tfds.load('mnist', split="train", as_supervised=False) # False -> Tuple; True -> Dict
# # train_dataset_for_adv_model = tfds.as_numpy(train_set)
# test_dataset_for_adv_model = tfds.load('mnist', split="test", as_supervised=False)
# # test_dataset_for_adv_model = tfds.as_numpy(test_dataset_for....)
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# # would non-iterable Tensor error come up if I process the evaluation dataset as a dict by default --> convert (x_test, y_test) into a dict (dict of dicts)
# x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
# x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# # method 2
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
# x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# standard dataset processed for base client
datasets = tfds.load('mnist')
train_dataset = datasets['train']
test_dataset = datasets['test']
# .as_numpy --> dict

adv_train = train_dataset.map(convert_to_dictionaries)
adv_test = test_dataset.map(convert_to_dictionaries)
print(type(adv_train),type(adv_test))

# need to fix processing mnist data to adv regularized model for training