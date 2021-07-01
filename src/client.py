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
from tensorflow.keras.callbacks import LearningRateScheduler
# todo: hardcode arguments from parse_args for both client, server, and experiment config
# todo: get the base AdvRegClient to work with fit_round and evaluate_round with federated server-client process
# todo: implement formal robustness property checks/formulations 

class HParams(object):
    '''
        adv_multiplier: The weight of adversarial loss in the training objective, relative to the labeled loss.
        adv_step_size: The magnitude of adversarial perturbation.
        adv_grad_norm: The norm to measure the magnitude of adversarial perturbation.
    '''

    def __init__(self, num_classes, adv_multiplier, adv_step_size, adv_grad_norm, **kwargs):
        # store model and its respective train/test dataset + metadata in parameters
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

class AdvRegClientConfig(object):
    def __init__(self, model : AdversarialRegularization, params : HParams, train_dataset, test_dataset, validation_steps, validation_split=0.1):
        # when we iteratively update params and the dataset in terms of the current client being sampled for fit_round and eval_round, the config simplifies accessing the variables' state
        self.model = model
        self.params = params
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_steps = validation_steps
        self.validation_split = validation_split

class ClientConfig(object):
    # difference in datasets given adv_model and base_model
    def __init__(self, model : tf.keras.models.Model, params : HParams, train_dataset, test_dataset, validation_steps, validation_split=0.1):
        self.model = model
        self.params = params
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_steps = validation_steps
        self.validation_split = validation_split

class ExperimentConfig(object):
    def __init__(self, client_config, params, args):
        # client config stores partition datasets and model to use for client, independent of model type (MapDataset/BatchDataset fit to both model types)
        self.client_config = client_config
        # params store (adv) hyperparameter configurations that will iteratively update given each .sh execution
        self.params = params
        # args object; usage ex: exp_config.args.num_clients; args stores all the experiment-specific configurations (based on all permutations) defined by the user; args passed in main also works
        self.args = args

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

def build_gaussian_base_model(params : HParams):
    if (params.gaussian_state):
        input_layer = layers.Input(shape=(28, 28, 1), batch_size=None, name="image")
        gaussian_layer = keras.layers.GaussianNoise(stddev=0.2)(input_layer)
        conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(gaussian_layer)
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
        return model

def build_gaussian_adv_model(params : HParams):
    # precondition matches state check
    if (params.gaussian_state):
        input_layer = layers.Input(shape=(28, 28, 1), batch_size=None, name="image")
        gaussian_layer = keras.layers.GaussianNoise(stddev=0.2)(input_layer)
        conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(gaussian_layer)
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

def setup_client_parse_args():
    parser = argparse.ArgumentParser(description="Crypton Client")
    # configurations
    parser.add_argument("--num_partitions", type=int, choices=range(0, 10), required=False)
    parser.add_argument("--adv_grad_norm", type=str, required=False)
    parser.add_argument("--adv_multiplier", type=float, required=False, default=0.2)
    parser.add_argument("--adv_step_size", type=float, choices=range(0, 1), required=False)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--epochs", type=int, required=False)
    parser.add_argument("--num_clients", type=int, required=False, default=10)
    parser.add_argument("--adv_reg", type=bool, required=False, default=True)
    parser.add_argument("--gaussian_layer", type=bool, required=False)
    parser.add_argument("--weight_regularization", type=bool, required=False)
    parser.add_argument("--sgd_momentum", type=float, required=False, default=0.9)
    return parser

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# setup models
params = HParams(num_classes=10, adv_multiplier=0.2, adv_step_size=0.05, adv_grad_norm="infinity")
base_model = build_base_model(params=params)
adv_model = build_adv_model(params=params)

# MapDataset for partition creation
datasets = tfds.load('mnist')
train_dataset = datasets['train']
test_dataset = datasets['test']
train_dataset_for_base_model = train_dataset.map(normalize).shuffle(10000).batch(params.batch_size).map(convert_to_tuples)
test_dataset_for_base_model = test_dataset.map(normalize).batch(params.batch_size).map(convert_to_tuples)

# type: MapDataset
train_set_for_adv_model = train_dataset_for_base_model.map(convert_to_dictionaries)
test_set_for_adv_model = test_dataset_for_base_model.map(convert_to_dictionaries)

# create adv_reg client partitions
adv_train_partitions = Data.create_train_partitions(train_set_for_adv_model, num_clients=10)
adv_test_partitions = Data.create_test_partitions(test_set_for_adv_model, num_clients=10)

# create client partitions
train_partitions = Data.create_train_partitions(train_dataset_for_base_model, num_clients=10)
test_partitions = Data.create_test_partitions(test_dataset_for_base_model, num_clients=10)

# prepare BatchDataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test, y_test = x_train[-10000:], y_train[-10000:]
x_train = tf.cast(x_train, dtype=tf.float32)
x_test = tf.cast(x_test, dtype=tf.float32)
val_steps = x_test.shape[0] / params.batch_size

# type: BatchDataset
train_data = tf.data.Dataset.from_tensor_slices({'image': x_train, 'label': y_train}).batch(params.batch_size)
val_data = tf.data.Dataset.from_tensor_slices({'image': x_test, 'label': y_test}).batch(params.batch_size)

# setup client configurations; hardcoded configs for now
adv_client_config = AdvRegClientConfig(model=adv_model, params=params, train_dataset=train_data, test_dataset=val_data, validation_steps=val_steps)
client_config = ClientConfig(model=base_model, params=params, train_dataset=train_data, test_dataset=val_data, validation_steps=val_steps)
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

class AdvRegClient(flwr.client.KerasClient):
    def get_parameters(self):
        return adv_client_config.model.get_weights()

    def fit(self, parameters, config):
        adv_client_config.model.set_weights(parameters)
        # validation_data=adv_client_config.test_dataset, validation_steps=adv_client_config.validation_steps, validation_split=0.1, epochs=1
        history = adv_client_config.model.fit(adv_client_config.train_dataset, steps_per_epoch=3, epochs=5)
        results = {
            "loss": history.history["loss"],
            "sparse_categorical_crossentropy": history.history["sparse_categorical_crossentropy"],
            "sparse_categorical_accuracy": history.history["sparse_categorical_accuracy"],
            "scaled_adversarial_loss": history.history["scaled_adversarial_loss"],
        }

        return adv_client_config.model.get_weights(), len(adv_client_config.train_dataset), results

    def evaluate(self, parameters, config):
        adv_client_config.model.set_weights(parameters)
        results = adv_client_config.model.evaluate(adv_client_config.test_dataset, verbose=1)
        # only fit uses validation accuracy and sce loss
        results = {
                "loss": results[0],
                "sparse_categorical_crossentropy": results[1],
                "sparse_categorical_accuracy": results[2],
                "scaled_adversarial_loss": results[3],
        }
        # or just history[0], history[1]
        return results["loss"], results["sparse_categorical_accuracy"]

class Client(flwr.client.KerasClient):
    def get_parameters(self):
        return client_config.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        # validation data param may be negligible; validation_data=client_config.test_dataset, validation_steps=client_config.validation_steps, validation_split=0.1, steps_per_epoch=3, epochs=1, verbose=1
        history = client_config.model.fit(client_config.train_dataset, steps_per_epoch=3, epochs=5, callbacks=[callback])
        # run the entire base model and check for its errors
        results = {
            "loss": history.history["loss"],
            "sparse_categorical_crossentropy": history.history["sparse_categorical_crossentropy"],
            "sparse_categorical_accuracy": history.history["sparse_categorical_accuracy"],
            "scaled_adversarial_loss": history.history["scaled_adversarial_loss"],
        }
        return client_config.model.get_weights(), len(client_config.train_dataset), results

    def evaluate(self, parameters, config):
        client_config.model.set_weights(parameters)
        # it's more abt the dataset types consistent with both nsl and flwr.client; test backwards from what works with client to nsl model
        results = client_config.model.evaluate(client_config.test_dataset, verbose=1)
        # get loss from result list/dict
        results = {
                "loss": results[0],
                "sparse_categorical_crossentropy": results[1],
                "sparse_categorical_accuracy": results[2],
                "scaled_adversarial_loss": results[3],
        }

        return results["loss"], results["sparse_categorical_accuracy"]

def main():
    adv_reg_client = AdvRegClient()
    flwr.client.start_keras_client(server_address="[::]:8080", client=adv_reg_client)

if __name__ == '__main__':
    # client_parser = setup_client_parse_args()
    # args = client_parser.parse_args()
    main()
