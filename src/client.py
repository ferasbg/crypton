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
from keras.callbacks import History, EarlyStopping

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class DatasetConfig(object):
    '''
        DatasetConfig --> AdvRegClientConfig --> AdvRegClient
        DatasetConfig --> ClientConfig --> Client
    '''
    def __init__(self, args):
        # load in a partition instead of the entire dataset
        (x_train, y_train) = self.load_train_partition(idx=args.client_partition_idx)
        (x_test, y_test) = self.load_test_partition(idx=args.client_partition_idx)

        if (args.corruption_name != ""):
            x_train = self.corrupt_train_partition(x_train, corruption_name=args.corruption_name)

        self.partitioned_train_dataset = tf.data.Dataset.from_tensor_slices({'image': x_train, 'label': y_train}).batch(args.batch_size)
        self.partitioned_test_dataset = tf.data.Dataset.from_tensor_slices({'image': x_test, 'label': y_test}).batch(args.batch_size)
        self.partitioned_val_steps = len(self.partitioned_test_dataset) / args.batch_size # steps_per_epoch for test dataset
        self.train_dataset_cardinality = len(x_train)
        self.test_dataset_cardinality = len(x_test)

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
    parser.add_argument("--batch_size", type=int, required=False, default=16)
    parser.add_argument("--epochs", type=int, required=False, default=1)
    # parser.add_argument("--steps_per_epoch", type=int, required=False, default=1) // num_examples/batch_size, which is very different given a partition
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

    # number of epochs. Every value most likely will be the same.
    epoch_cardinality = []
    # if the vectors change, then we can store the average during each fit iteration. Given that we are using the .fit() function approximately for 50 iterations over 100 batches during 1 .fit() iteration (polynomial time?). Either way, we get the first element, regardless of the set contains 1 element or not.
    epoch_level_accuracy = []
    epoch_level_loss = []

    class AdvRegClient(flwr.client.KerasClient):
        def get_weights(self):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            history = model.fit(dataset_config.partitioned_train_dataset, epochs=args.epochs)
            results = {
                "loss": history.history["loss"],
                "sparse_categorical_crossentropy": history.history["sparse_categorical_crossentropy"],
                "sparse_categorical_accuracy": history.history["sparse_categorical_accuracy"],
                "scaled_adversarial_loss": history.history["scaled_adversarial_loss"],
            }

            train_cardinality = len(dataset_config.partitioned_train_dataset)
            accuracy : int = 0
            acc_vector = results["sparse_categorical_accuracy"]
            for i in range(len(acc_vector)):
                accuracy+=acc_vector[i]
            
            accuracy = accuracy / (len(acc_vector))
            accuracy = int(accuracy)
            epoch_level_accuracy.append(accuracy)

            # get the total epochs during the eval round for this client
            epoch_cardinality.append(len(results["loss"]))
            return model.get_weights(), train_cardinality, accuracy

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            res = model.evaluate(dataset_config.partitioned_train_dataset, verbose=1)
            results = {
                    # let's assume these elements are sets.
                    "loss": res[0],
                    "sparse_categorical_crossentropy": res[1],
                    "sparse_categorical_accuracy": res[2],
                    "scaled_adversarial_loss": res[3],
            }

            # EvaluateRes requires 1 loss value. Average out the loss vector.
            loss = 0
            loss_vector = res.history["loss"]
            for k in range(len(loss_vector)):
                loss+=loss_vector[k]

            loss = loss / len(loss_vector)
            loss = int(loss)

            accuracy = 0
            # let's assume res.history["sca"] = results["sca"]
            acc_vector = res.history["sparse_categorical_accuracy"]
            for i in range(len(acc_vector)):
                accuracy+=acc_vector[i]
            
            accuracy = accuracy / (len(acc_vector))
            accuracy = int(accuracy)
            
            test_cardinality = len(dataset_config.partitioned_test_dataset)
            # get the total epochs during the eval round for this client
            loss_cardinality = len(res.history["loss"])
            epoch_cardinality.append(loss_cardinality)

            return loss, test_cardinality, accuracy

    class Client(flwr.client.KerasClient):
        def get_weights(self):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)

            history = model.fit(dataset_config.partitioned_train_dataset, validation_data=dataset_config.partitioned_test_dataset, validation_steps=dataset_config.partitioned_val_steps, epochs=args.epochs)
            results = {
                "loss": history.history["loss"],
                "sparse_categorical_crossentropy": history.history["sparse_categorical_crossentropy"],
                "sparse_categorical_accuracy": history.history["sparse_categorical_accuracy"],
                "scaled_adversarial_loss": history.history["scaled_adversarial_loss"],
            }

            train_cardinality = len(dataset_config.partitioned_train_dataset)

            accuracy : int = 0
            acc_vector = results["sparse_categorical_accuracy"]
            for i in range(len(acc_vector)):
                accuracy+=acc_vector[i]

            accuracy = int(accuracy)
            epoch_level_accuracy.append(accuracy)
            # epochs depends on loss cardinality..
            epoch_cardinality.append(len(results["loss"]))

            # flwr FitRes requires 1 accuracy value. 
            return model.get_weights(), train_cardinality, accuracy

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            results = model.evaluate(dataset_config.partitioned_test_dataset, verbose=1)
            epoch_cardinality.append(len(results.history['loss']))    
            results = {
                    "loss": results[0],
                    "sparse_categorical_crossentropy": results[1],
                    "sparse_categorical_accuracy": results[2],
                    "scaled_adversarial_loss": results[3],
            }

            loss = 0
            loss_vector = results.history["loss"]
            for k in range(len(loss_vector)):
                loss+=loss_vector[k]

            loss = loss / len(loss_vector)
            loss = int(loss)

            accuracy : int = 0
            acc_vector = results["sparse_categorical_accuracy"]
            for i in range(len(acc_vector)):
                accuracy+=acc_vector[i]

            accuracy = int(accuracy)
            test_cardinality = len(dataset_config.partitioned_test_dataset)

            return loss, test_cardinality, accuracy

    if (args.client == "base_client"):
        client = Client()

    if (args.client == "nsl_client"):
        client = AdvRegClient()

    else:
        client = Client()

    flwr.client.start_keras_client(server_address="[::]:8080", client=client)

    # The evaluate() function uses the test dataset and computes a regularization loss. Thus the more clients under fraction_fit, the better the client evaluation loss. 
    client_results : Dict = {
        # the value in this store would be the vector average of the epoch-level accuracies, that is stored already in epoch_level_accuracy
        "client_accuracy": epoch_level_accuracy[0],
        "client_regularization_loss": epoch_level_loss[0],
    }

    for k in range(len(epoch_level_accuracy)):
        print(epoch_level_accuracy[k])
    
    for l in range(len(epoch_level_loss)):
        print(epoch_level_loss[l])

    for epoch in epoch_cardinality:
        print(epoch + "\n")

    # todo: write in the epoch and the respective client and loss value inside the text file, thus getting you the data for the regularization loss / acc at the client-set level, not just individual clients.
    

    