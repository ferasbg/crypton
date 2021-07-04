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
from utils import *
from tensorflow.keras.callbacks import LearningRateScheduler
from flwr.server.strategy import FedAdagrad, FedAvg, FaultTolerantFedAvg, FedFSv1
import bokeh
import seaborn as sns
import pandas as pd
import chartify
import matplotlib.pyplot as plt

# todo: setup corruptions with DatasetConfig
# todo: test each corruptions func
# todo: add fedadagrad and faulttolerantfedavg once they can be supported and are functional
# todo: setup exp configs; hardcode the graphs (x-y axis) that will be made based on the notes you have in dynalist and write the pseudocode in terms of matplotlib.pyplot if necessary
# todo: write test_plot_creation_with_dummy_data_for_exp_config():
# todo: apply corruptions to feature tuples given args in DatasetConfig
# todo: create sample plots based on target plots required for paper, then add with data from running exp-config-run.sh

class AdvRegClientConfig(object):
    def __init__(self, model : AdversarialRegularization, params : HParams, train_dataset, test_dataset, validation_steps=None, validation_split=0.1):
        # when we iteratively update params and the dataset in terms of the current client being sampled for fit_round and eval_round, the config simplifies accessing the variables' state
        self.model = model
        self.params = params
        # precondition: Partitioned BatchDataset
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_steps = validation_steps
        self.validation_split = validation_split

class ClientConfig(object):
    # precondition 1: train_dataset is a partition given num_clients from DatasetConfig and ExperimentConfig
    # precondition 2: test_dataset is a partition given num_clients from DatasetConfig and ExperimentConfig
    def __init__(self, model : tf.keras.models.Model, params : HParams, train_dataset, test_dataset, validation_steps=None, validation_split=0.1):
        self.model = model
        self.params = params
        # precondition: Partitioned BatchDataset
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_steps = validation_steps
        self.validation_split = validation_split

# functions to convert tuple into BatchDataset to be processed in the client models
def normalize(features):
  features['image'] = tf.cast(
      features['image'], dtype=tf.float32) / 255.0
  return features

def convert_to_tuples(features):
  return features['image'], features['label']

def convert_to_dictionaries(image, label, IMAGE_INPUT_NAME='image', LABEL_INPUT_NAME='label'):
  return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}

class DatasetConfig(object):
    '''
        DatasetConfig --> AdvRegClientConfig --> AdvRegClient
        DatasetConfig --> ClientConfig --> Client
    '''
    def __init__(self, client_partition_idx : int):
        # server.py params for strategy map to the partitions that depend on num_clients
        
        # (x_train, y_train) = self.load_train_partition(idx=client_partition_idx)
        # (x_test, y_test) = self.load_test_partition(idx=client_partition_idx)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        self.train_data = tf.data.Dataset.from_tensor_slices({'image': x_train, 'label': y_train}).batch(32)
        self.val_data = tf.data.Dataset.from_tensor_slices({'image': x_test, 'label': y_test}).batch(32)
        # partition / batch size
        self.val_steps = 5 

    def load_partition(self, idx : int):
        assert idx in range(10)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = tf.cast(x_train, dtype=tf.float32)
        x_test = tf.cast(x_test, dtype=tf.float32)

        return (
            x_train[idx * 5000 : (idx + 1) * 5000],
            y_train[idx * 5000 : (idx + 1) * 5000],
        ), (
            x_test[idx * 1000 : (idx + 1) * 1000],
            y_test[idx * 1000 : (idx + 1) * 1000],
        )

    def load_train_partition(self, idx: int):
        # the declaration is in terms of a tuple to the assignment with the respective load partition function
        assert idx in range(10)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = tf.cast(x_train, dtype=tf.float32)
        x_test = tf.cast(x_test, dtype=tf.float32)

        # process the same dataset
        return (x_train[idx * 5000 : (idx + 1) * 5000], y_train[idx * 5000 : (idx + 1) * 5000])

    def load_test_partition(self, idx : int):
        assert idx in range(10)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = tf.cast(x_train, dtype=tf.float32)
        x_test = tf.cast(x_test, dtype=tf.float32)
        return (x_test[idx * 1000 : (idx + 1) * 1000], y_test[idx * 1000 : (idx + 1) * 1000])

    def corrupt_train_partition(self, corruption_name : str):
        '''
        Usage:
            client_train_partition = self.corrupt_train_partition(self.client_train_partition, corruption_name=args.corruption_name)
        '''
        # corrupt the dataset in DatasetConfig
        for i in range(len(self.x_train)):
            self.x_train[i] = imagecorruptions.corrupt(self.x_train[i], corruption_name=corruption_name)

class MetricsConfig(object):
    @staticmethod
    def create_table(header: list, csv, norm_type="l-inf", options=["l-inf", "l2"]):
        # every set of rounds maps to a hardcoded adv_step_size, so we can measure this round set in terms of the adv_step_size set we want to iterate over
        headers = ["Model", "Adversarial Regularization Technique", "Strategy", "Server Model ε-Robust Federated Accuracy", "Server Model Certified ε-Robust Federated Accuracy"]
        # define options per variable; adv reg shares pattern of adversarial augmentation, noise (non-uniform, uniform) perturbations/corruptions/degradation as regularization
        adv_reg_options = ["Neural Structured Learning", "Gaussian Regularization", "Data Corruption Regularization", "Noise Regularization", "Blur Regularization"]
        strategy_options = ["FedAvg", "FedAdagrad", "FaultTolerantFedAvg", "FedFSV1"]
        metrics = ["server_loss", "server_accuracy_under_attack", "server_certified_loss"]
        # norm type used to define the line graph in the plot rather than an x or y axis label
        variables = ["epochs", "communication_rounds", "client_learning_rate", "server_learning_rate", "adv_grad_norm", "adv_step_size"]
        nsl_variables = ["neighbor_loss"]
        # measure severity as an epsilon when labeling the line graph in plot
        baseline_adv_reg_variables = ["severity", "noise_sigma"]

    @staticmethod
    def plot_client_model(model):
        file = keras.utils.plot_model(model, to_file='model.png')
        save_path = '/media'
        file_name = "model.png"
        os.path.join(save_path, file_name)

    @staticmethod
    def plot_img(image : np.ndarray):
        plt.figure(figsize=(32,32))
        # iteratively get perturbed images based on their norm type and norm values (l∞-p_ε; norm_type, adv_step_size)
        plt.imshow(image, cmap=plt.get_cmap('gray'))

def build_base_model(params : HParams):
    input_layer = layers.Input(shape=(32,32,3), name="image")
    regularizer = tf.keras.regularizers.l2()
    conv1 = layers.Conv2D(32, (3,3), activation='relu', kernel_regularizer=regularizer, padding='same')(input_layer)
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
    input_layer = layers.Input(shape=(32,32,3), name="image")
    regularizer = tf.keras.regularizers.l2()
    conv1 = layers.Conv2D(32, (3,3), activation='relu', kernel_regularizer=regularizer, padding='same')(input_layer)
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
    input_layer = layers.Input(shape=(32,32,1), batch_size=None, name="image")
    regularizer = tf.keras.regularizers.l2()
    gaussian_layer = keras.layers.GaussianNoise(stddev=0.2)(input_layer)
    conv1 = layers.Conv2D(32, (3,3), activation='relu', kernel_regularizer=regularizer, padding='same')(gaussian_layer)
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
    input_layer = layers.Input(shape=(32,32,1), batch_size=None, name="image")
    regularizer = tf.keras.regularizers.l2()
    gaussian_layer = keras.layers.GaussianNoise(stddev=0.2)(input_layer)
    conv1 = layers.Conv2D(32, (3,3), activation='relu', kernel_regularizer=regularizer, padding='same')(gaussian_layer)
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
    # would y_train and y_test need to become .to_categorical(y_train, 10) or not?
    return adv_model

def build_base_sequential_model(params : HParams):
    # build layers of public neural network
    model = Sequential()
    # feature layers
    model.add(Conv2D(32, (3, 3), activation='relu',
                        kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='relu',
                        kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu',
                        kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu',
                        kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu',
                        kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu',
                        kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    # classification layers
    model.add(Dense(128, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(params.num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros'))
        # stochastic gd has momentum, optimizer doesn't use momentum for weight regularization
    # optimizer = Adam(learning_rate=0.001)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def setup_client_parse_args():
    parser = argparse.ArgumentParser(description="Crypton Client")
    # configurations
    parser.add_argument("--client_partition_idx", type=int, required=False, default=0)
    parser.add_argument("--adv_grad_norm", type=str, required=False, default="infinity")
    parser.add_argument("--adv_multiplier", type=float, required=False, default=0.2)
    parser.add_argument("--adv_step_size", type=float, required=False, default=0.05)
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--epochs", type=int, required=False, default=1)
    parser.add_argument("--steps_per_epoch", type=int, required=False, default=0)
    parser.add_argument("--num_clients", type=int, required=False, default=10)
    parser.add_argument("--num_classes", type=int, required=False, default=10)
    parser.add_argument("--nsl_reg", type=bool, required=False, default=True)
    parser.add_argument("--gaussian_reg", type=bool, required=False, default=False)
    parser.add_argument("--corruption_name", type=str, required=False, default="jpeg_compression")
    parser.add_argument("--data_corruption_reg", type=str, required=False, default="jpeg_compression")
    parser.add_argument("--noise_corruption_reg", type=str, required=False, default="shot_noise")
    parser.add_argument("--blur_corruption_reg", type=str, required=False, default="motion_blur")
    return parser

if __name__ == '__main__':
    # process args upon execution
    client_parser = setup_client_parse_args()
    args = client_parser.parse_args()
    dataset_config = DatasetConfig(args.client_partition_idx)
    params = HParams(num_classes=args.num_classes, adv_multiplier=args.adv_multiplier, adv_step_size=args.adv_step_size, adv_grad_norm=args.adv_grad_norm)
    
    nsl_model = build_adv_model(params=params)
    base_model = build_base_model(params=params)
    gaussian_base_model = build_gaussian_base_model(params=params)
    gaussian_adv_model = build_gaussian_adv_model(params=params)

    # datasets have been partitioned per client
    adv_client_config = AdvRegClientConfig(model=nsl_model, params=params, train_dataset=dataset_config.train_data, test_dataset=dataset_config.val_data)
    client_config = ClientConfig(model=base_model, params=params, train_dataset=dataset_config.train_data, test_dataset=dataset_config.val_data, validation_steps=dataset_config.val_steps)

    class AdvRegClient(flwr.client.KerasClient):
        def get_weights(self):
            return adv_client_config.model.get_weights()

        def fit(self, parameters, config):
            adv_client_config.model.set_weights(parameters)
            history = adv_client_config.model.fit(dataset_config.train_data, validation_data=dataset_config.val_data, validation_steps=dataset_config.val_steps, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs)
            results = {
                "loss": history.history["loss"],
                "sparse_categorical_crossentropy": history.history["sparse_categorical_crossentropy"],
                "sparse_categorical_accuracy": history.history["sparse_categorical_accuracy"],
                "scaled_adversarial_loss": history.history["scaled_adversarial_loss"],
            }

            # adv_client_config.train_dataset --> depends on dataset_config.train_data == train_data
            train_cardinality = len(dataset_config.train_data)
            accuracy = results["sparse_categorical_accuracy"]
            accuracy = int(accuracy[0])
            return adv_client_config.model.get_weights(), train_cardinality, accuracy

        def evaluate(self, parameters, config):
            adv_client_config.model.set_weights(parameters)
            results = adv_client_config.model.evaluate(dataset_config.val_data, verbose=1)
            # only fit uses validation accuracy and sce loss
            results = {
                    "loss": results[0],
                    "sparse_categorical_crossentropy": results[1],
                    "sparse_categorical_accuracy": results[2],
                    "scaled_adversarial_loss": results[3],
            }

            loss = int(results["loss"])
            accuracy = int(results["sparse_categorical_accuracy"])
            # client_configs[0].test_dataset
            test_cardinality = len(dataset_config.val_data)

            return loss, test_cardinality, accuracy

    class Client(flwr.client.KerasClient):
        def get_weights(self):
            return client_config.model.get_weights()

        def fit(self, parameters, config):
            client_config.model.set_weights(parameters)
            history = client_config.model.fit(dataset_config.train_data, validation_data=dataset_config.val_data, validation_steps=dataset_config.val_steps, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs)
            results = {
                "loss": history.history["loss"],
                "sparse_categorical_crossentropy": history.history["sparse_categorical_crossentropy"],
                "sparse_categorical_accuracy": history.history["sparse_categorical_accuracy"],
                "scaled_adversarial_loss": history.history["scaled_adversarial_loss"],
            }

            train_cardinality = len(dataset_config.train_data)
            accuracy = results["sparse_categorical_accuracy"]
            accuracy = int(accuracy[0])

            return client_config.model.get_weights(), train_cardinality, accuracy

        def evaluate(self, parameters, config):
            client_config.model.set_weights(parameters)
            results = client_config.model.evaluate(dataset_config.val_data, verbose=1)
            results = {
                    "loss": results[0],
                    "sparse_categorical_crossentropy": results[1],
                    "sparse_categorical_accuracy": results[2],
                    "scaled_adversarial_loss": results[3],
            }

            loss = int(results["loss"])
            accuracy = int(results["sparse_categorical_accuracy"])
            test_cardinality = len(dataset_config.val_data)

            return loss, test_cardinality, accuracy

    nsl_client = AdvRegClient()
    client = Client()
    flwr.client.start_keras_client(server_address="[::]:8080", client=client)
