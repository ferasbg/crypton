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

datasets = tfds.load('mnist')
train_dataset = datasets['train']
test_dataset = datasets['test']
# tuple needs features, dict needs split features
train_dataset_for_base_model = train_dataset.map(normalize).shuffle(10000).batch(batch_size=32).map(convert_to_tuples)
test_dataset_for_base_model = test_dataset.map(normalize).batch(batch_size=32).map(convert_to_tuples)

train_dataset_for_adv_model = tfds.load('mnist', split="train") # False -> Tuple; True -> Dict
test_dataset_for_adv_model = tfds.load('mnist', split="test")
print(len(train_dataset_for_base_model))
print(len(test_dataset_for_adv_model))
# PrefetchDataset
print(type(train_dataset_for_adv_model), type(test_dataset_for_adv_model))

# IterableDataset --> convert to iterable dict of np.ndarrays

train = tfds.as_numpy(train_dataset_for_adv_model)
test = tfds.as_numpy(test_dataset_for_adv_model)
print(type(train), type(test))

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

# reshape the data above before processing with function below
# x_train = x_train.reshape((-1, 28, 28, 1))
# x_test = x_test.reshape((-1, 28, 28, 1))
# y_train = tf.keras.utils.to_categorical(y_train, 10)
# y_test = tf.keras.utils.to_categorical(y_test, 10)


# adv_train = train_dataset.map(normalize).shuffle(10000).batch(params.batch_size).map(convert_to_dictionaries)
# adv_test = test_dataset.map(normalize).batch(params.batch_size).map(convert_to_dictionaries)

# train_dataset_for_adv_model = tfds.load('mnist', split="train", as_supervised=True) # False -> Tuple; True -> Dict
# test_dataset_for_adv_model = tfds.load('mnist', split="test", as_supervised=True)
# would I need to use different loss func if I don't reshape?


## optimizations after base
    # todo: create __init__ func and add args param to configure AdvRegClient and Client
