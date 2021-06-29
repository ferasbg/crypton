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

IMAGE_INPUT_NAME = 'image'
LABEL_INPUT_NAME = 'label'
datasets = tfds.load('mnist')
train_dataset = datasets['train']
test_dataset = datasets['test']
train_dataset_for_base_model = train_dataset.map(normalize).shuffle(10000).batch(batch_size=32).map(convert_to_tuples)
test_dataset_for_base_model = test_dataset.map(normalize).batch(batch_size=32).map(convert_to_tuples)

# adv_model needs it to be processed in a dict, but flwr.client processes it as a tuple; this conflict is my error
train_set_for_adv_model = train_dataset_for_base_model.map(convert_to_dictionaries)
test_set_for_adv_model = test_dataset_for_base_model.map(convert_to_dictionaries)
# both datasets for the base and adv model should be of the same MapDataset type independent of adv_model feature dicts
print(type(train_set_for_adv_model), type(test_set_for_adv_model))

# later for AdvRegClient
  # self.train_dataset = args.current_train_partition
        # self.test_dataset = args.current_train_partition
        # # for using different models, try self.model = args.model
        # # store gaussian layer to append to existing model if True
        # self.params = args.params

# define train/test partition and client partitions
(x_train, y_train) = tf.keras.datasets.mnist.load_data()
x_test, y_test = x_train[45000:50000], y_train[45000:50000]
# they are all tuples that are iterable
print(type(x_train), type(y_train), type(x_test), type(y_test))
