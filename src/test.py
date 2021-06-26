import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

def normalize(features):
  features['image'] = tf.cast(
      features['image'], dtype=tf.float32) / 255.0
  return features

def convert_to_tuples(features):
  return features['image'], features['label']

def convert_to_dictionaries(image, label):
  return {'image': image, 'label': label}

# iteratively process data with convert_to_dictionaries: 1 method for processing adv_reg data actually may work: convert existing callable function iteratively on dataset instead to have a dict of numpy arrays

# standard dataset processed for base client
datasets = tfds.load('mnist', as_supervised=True)
train_dataset = datasets['train']
test_dataset = datasets['test']
train_dataset_for_base_model = train_dataset.map(normalize).shuffle(10000).batch(batch_size=32).map(convert_to_tuples)
test_dataset_for_base_model = test_dataset.map(normalize).batch(batch_size=32).map(convert_to_tuples)

train_set = tfds.load('mnist', split="train", as_supervised=False) # False -> Tuple; True -> Dict
train_dataset_for_adv_model = tfds.as_numpy(train_set)
print(type(train_dataset_for_adv_model))
test_set = tfds.load('mnist', split="test", as_supervised=False)
test_dataset_for_adv_model = tfds.as_numpy(test_set)
print(type(test_dataset_for_adv_model))

train_set = tfds.load('mnist', split="train", as_supervised=True) # False -> Tuple; True -> Dict
train_dataset_for_adv_model = tfds.as_numpy(train_set)
print(type(train_dataset_for_adv_model))


