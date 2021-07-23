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
import json
import jsonify
import logging
from logging import Logger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

## GET DATA FOR:
# server eval accuracy against comm rounds
# client train accuracy vs num rounds (for each adv_reg technique so constant adv_grad_norm and adv_step_size but different adv_reg techniques)
# server eval loss against comm rounds
# client train loss vs num rounds (for each adv_reg technique with constant adv_grad_norm and adv_step_size)

'''
# change to get all plot data
python3 client.py --steps_per_epoch=1 --epochs=1 --model="nsl_model" --num_clients=10  -adv_grad_norm="infinity" --adv_step_size=0.05 --client_partition_idx=0

INFO flower 2021-07-05 23:49:57,927 | app.py:121 | app_evaluate: federated loss: 625.0
INFO:flower:app_evaluate: federated loss: 625.0
INFO flower 2021-07-05 23:49:57,927 | app.py:122 | app_evaluate: results [('ipv6:[::1]:58558', EvaluateRes(loss=625.0, num_examples=48, accuracy=0.0, metrics={})), ('ipv6:[::1]:58564', EvaluateRes(loss=625.0, num_examples=47, accuracy=0.0, metrics={})), ('ipv6:[::1]:58568', EvaluateRes(loss=625.0, num_examples=47, accuracy=0.0, metrics={})), ('ipv6:[::1]:58572', EvaluateRes(loss=625.0, num_examples=48, accuracy=0.0, metrics={})), ('ipv6:[::1]:58574', EvaluateRes(loss=625.0, num_examples=47, accuracy=0.0, metrics={})), ('ipv6:[::1]:58578', EvaluateRes(loss=625.0, num_examples=48, accuracy=0.0, metrics={})), ('ipv6:[::1]:58582', EvaluateRes(loss=625.0, num_examples=48, accuracy=0.0, metrics={})), ('ipv6:[::1]:58586', EvaluateRes(loss=625.0, num_examples=47, accuracy=0.0, metrics={})), ('ipv6:[::1]:58590', EvaluateRes(loss=625.0, num_examples=49, accuracy=0.0, metrics={}))]
INFO:flower:app_evaluate: results [('ipv6:[::1]:58558', EvaluateRes(loss=625.0, num_examples=48, accuracy=0.0, metrics={})), ('ipv6:[::1]:58564', EvaluateRes(loss=625.0, num_examples=47, accuracy=0.0, metrics={})), ('ipv6:[::1]:58568', EvaluateRes(loss=625.0, num_examples=47, accuracy=0.0, metrics={})), ('ipv6:[::1]:58572', EvaluateRes(loss=625.0, num_examples=48, accuracy=0.0, metrics={})), ('ipv6:[::1]:58574', EvaluateRes(loss=625.0, num_examples=47, accuracy=0.0, metrics={})), ('ipv6:[::1]:58578', EvaluateRes(loss=625.0, num_examples=48, accuracy=0.0, metrics={})), ('ipv6:[::1]:58582', EvaluateRes(loss=625.0, num_examples=48, accuracy=0.0, metrics={})), ('ipv6:[::1]:58586', EvaluateRes(loss=625.0, num_examples=47, accuracy=0.0, metrics={})), ('ipv6:[::1]:58590', EvaluateRes(loss=625.0, num_examples=49, accuracy=0.0, metrics={}))]
INFO flower 2021-07-05 23:49:57,928 | app.py:127 | app_evaluate: failures []
INFO:flower:app_evaluate: failures []

# each client's accuracy/loss set is based on the num_rounds parameter; 
# since we are running multiple clients (but not in parallel), we write to the logfile such that we can construct the plot easily 
# it's a lot easier to read the txt file to create plots after iterations are complete

'''

class DatasetConfig(object):
    '''
        DatasetConfig --> AdvRegClientConfig --> AdvRegClient
        DatasetConfig --> ClientConfig --> Client
    '''
    def __init__(self, args):
        (x_train, y_train) = self.load_train_partition(idx=args.client_partition_idx)
        (x_test, y_test) = self.load_test_partition(idx=args.client_partition_idx)

        # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        if (args.corruption_name != ""):
            x_train = self.corrupt_train_partition(x_train, corruption_name=args.corruption_name)

        self.partitioned_train_dataset = tf.data.Dataset.from_tensor_slices({'image': x_train, 'label': y_train}).batch(args.batch_size)
        self.partitioned_test_dataset = tf.data.Dataset.from_tensor_slices({'image': x_test, 'label': y_test}).batch(args.batch_size)
        self.partitioned_val_steps = len(self.partitioned_test_dataset) / args.batch_size

    def load_train_partition(self, idx: int):
        # the declaration is in terms of a tuple to the assignment with the respective load partition function
        assert idx in range(10)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        # process the same dataset
        return (x_train[idx * 5000 : (idx + 1) * 5000], y_train[idx * 5000 : (idx + 1) * 5000])

    def load_test_partition(self, idx : int):
        assert idx in range(10)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        return (x_test[idx * 1000 : (idx + 1) * 1000], y_test[idx * 1000 : (idx + 1) * 1000])

    def corrupt_train_partition(self, x_train, corruption_name : str):
        '''
        Technically, the corruption is applied before it's partitioned.

        Usage:
            client_train_partition = self.corrupt_train_partition(self.client_train_partition, corruption_name=args.corruption_name)
        '''
        # corrupt the dataset in DatasetConfig
        for i in range(len(x_train)):
            x_train[i] = imagecorruptions.corrupt(x_train[i], corruption_name=corruption_name)

        return x_train

def build_base_model(params : HParams):
    input_layer = layers.Input(shape=[32,32,3], batch_size=None, name="image")
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
    input_layer = layers.Input(shape=(32,32,3), batch_size=None, name="image")
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

def setup_client_parser():
    parser = argparse.ArgumentParser(description="Crypton Client")
    # configurations
    parser.add_argument("--client_partition_idx", type=int, choices=range(0, 10), required=False, default=1)
    parser.add_argument("--adv_grad_norm", type=str, required=False, default="infinity")
    parser.add_argument("--adv_multiplier", type=float, required=False, default=0.2)
    parser.add_argument("--adv_step_size", type=float, required=False, default=0.05)
    # sample data is very sparse
    parser.add_argument("--batch_size", type=int, required=False, default=8)
    parser.add_argument("--epochs", type=int, required=False, default=1)
    parser.add_argument("--steps_per_epoch", type=int, required=False, default=1)
    parser.add_argument("--num_clients", type=int, required=False, default=10)
    parser.add_argument("--num_classes", type=int, required=False, default=10)
    # hardcode use of nsl model
    parser.add_argument("--model", type=str, required=False, default="nsl_model")
    parser.add_argument("--nsl_reg", type=bool, required=False, default=True)
    parser.add_argument("--gaussian_reg", type=bool, required=False, default=False)
    # corruptions act as perturbations and degradations, and nest noise/blur/data corruptions. Each unique corruption stands as an adversarial regularization technique.
    parser.add_argument("--corruption_name", type=str, required=False, default="")
    parser.add_argument("--nominal_reg", type=str, required=False, default=True)
    # options: base_client, nsl_client
    parser.add_argument("--client", type=str, required=False, default="nsl_client")
    parser = parser.parse_args()
    return parser

if __name__ == '__main__':
    # process args upon execution
    args = setup_client_parser()
    dataset_config = DatasetConfig(args)

    params = HParams(num_classes=args.num_classes, adv_multiplier=args.adv_multiplier, adv_step_size=args.adv_step_size, adv_grad_norm=args.adv_grad_norm)
    nsl_model = build_adv_model(params=params)
    base_model = build_base_model(params=params)
    gaussian_base_model = build_gaussian_base_model(params=params)

    # select model
    if (args.model == "nsl_model"):
        model = nsl_model

    if (args.model == "gaussian_model"):
        model = gaussian_base_model

    if (args.model == "base_model"):
        model = base_model

    client_accuracies = []
    client_losses = []

    class AdvRegClient(flwr.client.KerasClient):
        def get_weights(self):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            history = model.fit(dataset_config.partitioned_train_dataset, validation_data=dataset_config.partitioned_test_dataset, validation_steps=dataset_config.partitioned_val_steps, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs)
            results = {
                "loss": history.history["loss"],
                "sparse_categorical_crossentropy": history.history["sparse_categorical_crossentropy"],
                "sparse_categorical_accuracy": history.history["sparse_categorical_accuracy"],
                "scaled_adversarial_loss": history.history["scaled_adversarial_loss"],
            }

            train_cardinality = len(dataset_config.partitioned_train_dataset)
            accuracy = results["sparse_categorical_accuracy"]
            accuracy = int(accuracy[0])
            
            # add client loss/acc for plot data
            client_accuracies.append(accuracy)
            client_losses.append(int(results["loss"][0]))

            return model.get_weights(), train_cardinality, accuracy

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            results = model.evaluate(dataset_config.partitioned_train_dataset, verbose=1)
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
            test_cardinality = len(dataset_config.partitioned_train_dataset)

            return loss, test_cardinality, accuracy

    class Client(flwr.client.KerasClient):
        def get_weights(self):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            history = model.fit(dataset_config.partitioned_train_dataset, validation_data=dataset_config.partitioned_test_dataset, validation_steps=dataset_config.partitioned_val_steps, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs)
            results = {
                "loss": history.history["loss"],
                "sparse_categorical_crossentropy": history.history["sparse_categorical_crossentropy"],
                "sparse_categorical_accuracy": history.history["sparse_categorical_accuracy"],
                "scaled_adversarial_loss": history.history["scaled_adversarial_loss"],
            }

            train_cardinality = len(dataset_config.partitioned_train_dataset)
            accuracy = results["sparse_categorical_accuracy"]
            accuracy = int(accuracy[0])

            # add client loss/acc for plot data
            client_accuracies.append(accuracy)
            client_losses.append(int(results["loss"][0]))

            return model.get_weights(), train_cardinality, accuracy

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            results = model.evaluate(dataset_config.partitioned_test_dataset, verbose=1)
            results = {
                    "loss": results[0],
                    "sparse_categorical_crossentropy": results[1],
                    "sparse_categorical_accuracy": results[2],
                    "scaled_adversarial_loss": results[3],
            }

            loss = int(results["loss"])
            accuracy = int(results["sparse_categorical_accuracy"])
            test_cardinality = len(dataset_config.partitioned_test_dataset)

            return loss, test_cardinality, accuracy

    if (args.client == "base_client"):
        client = Client()

    if (args.client == "nsl_client"):
        client = AdvRegClient()

    else:
        client = Client()
    
    flwr.client.start_keras_client(server_address="[::]:8080", client=client)

    with open('./metrics/client_accuracy.txt', 'w') as client_accuracy:
        for round_wise_accuracy in client_accuracies:
            client_accuracy.write("{}\n".format(round_wise_accuracy))

    with open('./metrics/client_losses.txt') as client_losses:
        for round_wise_loss in client_losses:
            client_losses.write("{}\n".format(round_wise_loss))
