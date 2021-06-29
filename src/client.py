import argparse
import flwr
from neural_structured_learning.keras.adversarial_regularization import AdversarialRegularization
import tensorflow as tf
from tensorflow import keras
from keras import layers
import neural_structured_learning as nsl
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
from keras.regularizers import l2
import numpy as np
from dataset import *
from attacks import *

# todo: use LearningRateScheduler to configure client and server learning rate
# todo: add HParams object as argument for each client to configure the client from args

class HParams(object):
    '''
    adv_multiplier: The weight of adversarial loss in the training objective, relative to the labeled loss.
    adv_step_size: The magnitude of adversarial perturbation.
    adv_grad_norm: The norm to measure the magnitude of adversarial perturbation.

    Notes:
        - there are different regularization techniques, but keep technique constant
        - formalize relationship between adversarial input generation for a client and its server-side evaluation-under-attack.
        - adversarial regularization is very useful for defining an explicit structure e.g. structural signals rather than single samples.
        - nsl-ar structured signals provides more fine-grained information not available in feature inputs.
        - We can assume training with robust adversarial examples makes it robust against adversarial perturbations during inference (eval), but how does this process fair when there's a privacy-specific step of using client models to locally train on this data and use a federated optimization technique for server-side evaluation? How can we utilize unsupervised/semi-supervised learning and these "structured signals" to learn hidden representations in perturbed or otherwise corrupted data (image transformations) with applied gaussian noise (these configurations exist to simulate a real-world scenario). We want to then formalize this phenomenon and the results between each experimental configuration.
        - adv reg. --> how does this affect fed optimizer (regularized against adversarial attacks) and how would differences in fed optimizer affect adv. reg model? Seems like FedAdagrad is better on het. data, so if it was regularized anyway with adv. perturbation attacks, it should perform well against any uniform of non-uniform or non-bounded real-world or fixed norm perturbations.
        - wrap the adversarial regularization model to train under two other conditions relating to GaussianNoise and specified perturbation attacks during training specifically.
        - graph the feature representation given graph with respect to the graph of the rest of its computations, and the trusted aggregator eval
    '''

    def __init__(self, num_classes, adv_multiplier, adv_step_size, adv_grad_norm):
        self.input_shape = [28, 28, 1]
        self.num_classes = num_classes
        self.conv_filters = [32, 32, 64, 64, 128, 128, 256]
        self.kernel_size = (3, 3)
        self.pool_size = (2, 2)
        self.num_fc_units = [64]
        self.batch_size = 32
        self.epochs = 5
        self.adv_multiplier = adv_multiplier
        self.adv_step_size = adv_step_size
        self.adv_grad_norm = adv_grad_norm  # "l2" or "infinity" = l2_clip_norm if "l2"
        self.gaussian_state : bool = False
        # if (params.gaussian_state): model = tf.keras.models.Model.add(params.gaussian_layer, stack_index=1)
        self.gaussian_layer = keras.layers.GaussianNoise(stddev=0.2)
        self.clip_value_min = 0.0
        self.clip_value_max = 1.0

def build_base_model(params : HParams):
    input_layer = layers.Input(shape=(28, 28, 1), batch_size=None, name="image")
    conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
    batch_norm = layers.BatchNormalization()(conv1)
    dropout = layers.Dropout(0.3)(batch_norm)
    conv2 = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform',  padding='same')(dropout)
    maxpool1 = layers.MaxPool2D((2,2))(conv2)
    conv3 = layers.Conv2D(64, (3,3), activation='relu',  kernel_initializer='he_uniform', padding='same')(maxpool1)
    conv4 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv3)
    maxpool2 = layers.MaxPool2D((2,2))(conv4)
    conv5 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(maxpool2)
    conv6 = layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv5)
    maxpool3 = layers.MaxPool2D((2,2))(conv6)
    flatten = layers.Flatten()(maxpool3)
    # flatten is creating error because type : NoneType
    dense1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flatten)
    # possibly remove defined kernel/bias initializer, but functional API will check for this and removes error before processing model architecture and config
    output_layer = layers.Dense(params.num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')(dense1)
    model = keras.Model(inputs=input_layer, outputs=output_layer, name='client_model')
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model

def build_adv_model(params : HParams):
    input_layer = layers.Input(shape=(28, 28, 1), batch_size=None, name="image")
    conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
    batch_norm = layers.BatchNormalization()(conv1)
    dropout = layers.Dropout(0.3)(batch_norm)
    #  kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)
    conv2 = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform',padding='same')(dropout)
    maxpool1 = layers.MaxPool2D((2,2))(conv2)
    conv3 = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform',  padding='same')(maxpool1)
    conv4 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform',  padding='same')(conv3)
    maxpool2 = layers.MaxPool2D((2,2))(conv4)
    conv5 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform',  padding='same')(maxpool2)
    conv6 = layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_uniform',  padding='same')(conv5)
    maxpool3 = layers.MaxPool2D((2,2))(conv6)
    flatten = layers.Flatten()(maxpool3)
    dense1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flatten)
    # possibly remove defined kernel/bias initializer, but functional API will check for this and removes error before processing model architecture and config
    output_layer = layers.Dense(params.num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')(dense1)
    model = keras.Model(inputs=input_layer, outputs=output_layer, name='client_model')
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    adv_config = nsl.configs.make_adv_reg_config(multiplier=params.adv_multiplier, adv_step_size=params.adv_step_size, adv_grad_norm=params.adv_grad_norm)
    # AdvRegularization is a sub-class of tf.keras.Model, but it processes dicts instead for train and eval because of its decomposition approach for nsl
    adv_model = nsl.keras.AdversarialRegularization(model, label_keys=['label'], adv_config=adv_config, base_with_labels_in_features=True)
    adv_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return adv_model

def normalize(features):
  features['image'] = tf.cast(
      features['image'], dtype=tf.float32) / 255.0
  return features

def convert_to_tuples(features):
  return features['image'], features['label']

def convert_to_dictionaries(image, label, IMAGE_INPUT_NAME='image', LABEL_INPUT_NAME='label'):
  return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}

def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    datasets = tfds.load('mnist')
    # from the train split, can we partition the data?
    train_dataset = datasets['train']
    test_dataset = datasets['test']

    return (
        x_train[idx * 5000 : (idx + 1) * 5000],
        y_train[idx * 5000 : (idx + 1) * 5000],
    ), (
        x_test[idx * 1000 : (idx + 1) * 1000],
        y_test[idx * 1000 : (idx + 1) * 1000],
    )

# create_adv_client()
class AdvRegClient(flwr.client.KerasClient):
    def __init__(self, model : AdversarialRegularization, train_dataset, test_dataset, validation_steps, args=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_steps = validation_steps
        # store args and params : HParams object

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        results = self.model.fit(self.train_dataset, validation_data=self.test_dataset, validation_steps=self.validation_steps, validation_split=0.1, epochs=1, steps_per_epoch=3, verbose=1)
        results = {
                "loss": results[0],
                "sparse_categorical_crossentropy": results[1],
                "sparse_categorical_accuracy": results[2],
                "scaled_adversarial_loss": results[3],
        }
        # you could make a dict from history callback object
        return self.model.get_weights(), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        # unpack error because loading into a list is the correct approach, except python infers the type to store the function to the right at runtime
        history = self.model.evaluate(self.test_dataset, verbose=1)
        results = {
                "loss": history[0],
                "sparse_categorical_crossentropy": history[1],
                "sparse_categorical_accuracy": history[2],
                "scaled_adversarial_loss": history[3],
        }
        # or just history[0], history[1]
        return results["loss"], results["sparse_categorical_accuracy"]

class Client(flwr.client.KerasClient):
    def __init__(self, model : tf.keras.models.Model, train_dataset, test_dataset, validation_steps):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_steps = validation_steps

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        # validation data param may be negligible
        results = self.model.fit(self.train_dataset, validation_data=self.test_dataset, validation_steps=self.validation_steps, validation_split=0.1, steps_per_epoch=3, epochs=5, verbose=1)
        # run the entire base model and check for its errors
        results = {
                "loss": results[0],
                "sparse_categorical_crossentropy": results[1],
                "sparse_categorical_accuracy": results[2],
                "scaled_adversarial_loss": results[3],
        }
        return self.model.get_weights(), results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        # it's more abt the dataset types consistent with both nsl and flwr.client; test backwards from what works with client to nsl model
        results = self.model.evaluate(self.test_dataset, verbose=1)
        # get loss from result list/dict
        results = {
                "loss": results[0],
                "sparse_categorical_crossentropy": results[1],
                "sparse_categorical_accuracy": results[2],
                "scaled_adversarial_loss": results[3],
        }

        # return a tuple of loss, accuracy
        return results["loss"], results["sparse_categorical_accuracy"]

def main(args):
    # create models
    params = HParams(num_classes=10, adv_multiplier=0.2, adv_step_size=0.05, adv_grad_norm="infinity") # args.adv_step_size, args.adv_grad_norm
    adv_model = build_adv_model(params=params)
    base_model = build_base_model(params=params)
    model = adv_model

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test, y_test = x_train[-10000:], y_train[-10000:]
    x_train = tf.cast(x_train, dtype=tf.float32)
    x_test = tf.cast(x_test, dtype=tf.float32)
    val_steps = x_test.shape[0] / params.batch_size

    # MapDataset
    datasets = tfds.load('mnist')
    train_dataset = datasets['train']
    test_dataset = datasets['test']
    train_dataset_for_base_model = train_dataset.map(normalize).shuffle(10000).batch(params.batch_size).map(convert_to_tuples)
    test_dataset_for_base_model = test_dataset.map(normalize).batch(params.batch_size).map(convert_to_tuples)

    # type: MapDataset
    train_set_for_adv_model = train_dataset_for_base_model.map(convert_to_dictionaries)
    test_set_for_adv_model = test_dataset_for_base_model.map(convert_to_dictionaries)

    # type: BatchDataset
    train_data = tf.data.Dataset.from_tensor_slices({'image': x_train, 'label': y_train}).batch(params.batch_size)
    val_data = tf.data.Dataset.from_tensor_slices({'image': x_test, 'label': y_test}).batch(params.batch_size)

    # start_client()
    if (type(model) == AdversarialRegularization):
        adv_reg_client = AdvRegClient(model, train_dataset=train_data, test_dataset=val_data, validation_steps=val_steps)
        flwr.client.start_keras_client(server_address="[::]:8080", client=adv_reg_client)

    elif (type(model) == tf.keras.models.Model):
        base_client = Client(model, train_dataset=train_dataset_for_base_model, test_dataset=test_dataset_for_base_model, validation_steps=val_steps)
        flwr.client.start_keras_client("[::]:8080", client=base_client)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crypton Client")
    # configurations
    args = parser.parse_args()
    parser.add_argument("--num_partitions", type=int, choices=range(0, 10), required=False)
    parser.add_argument("--adv_grad_norm", type=str, required=False)
    parser.add_argument("--adv_multiplier", type=float, required=False, default=0.2)
    parser.add_argument("--adv_step_size", type=float, choices=range(0, 1), required=False)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--epochs", type=int, required=False)
    parser.add_argument("--num_clients", type=int, required=False)
    parser.add_argument("--adv_reg", type=bool, required=False)
    parser.add_argument("--gaussian_layer", type=bool, required=False)
    parser.add_argument("--weight_regularization", type=bool, required=False)
    parser.add_argument("--sgd_momentum", type=float, required=False, default=0.9)
    main(args)

