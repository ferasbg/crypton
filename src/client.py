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

# todo: add support for the corruptions (as baseline adversarial regularization technique)
# todo: test all corruption functions in utils.Data
# todo: setup exp configs; hardcode the graphs (x-y axis) that will be made based on the notes you have in dynalist and write the pseudocode in terms of matplotlib.pyplot if necessary

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
    # precondition 1: train_dataset is a partition given num_clients from DatasetConfig and ExperimentConfig
    # precondition 2: test_dataset is a partition given num_clients from DatasetConfig and ExperimentConfig
    def __init__(self, model : tf.keras.models.Model, params : HParams, train_dataset, test_dataset, validation_steps, validation_split=0.1):
        self.model = model
        self.params = params
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
    def __init__(self, client_partition_idx : int, args):
        
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = tf.cast(x_train, dtype=tf.float32)
        x_test = tf.cast(x_test, dtype=tf.float32)
        self.val_steps = x_test.shape[0] / 32

        # partition data before passing into BatchDataset object
        if (args.num_clients in range(10) and args.num_clients < 11):
            self.x_train, self.y_train = self.load_train_partition_for_10_clients(idx=client_partition_idx)
            self.x_test, self.y_test = self.load_test_partition_for_10_clients(idx=client_partition_idx)
            # train_data that processes the BatchDataset using x_train and y_train in DatasetConfig
        
        if (args.num_clients in range(100) and args.num_clients > 10):
            (self.x_train, self.y_train) = self.load_train_partition_for_100_clients(idx=client_partition_idx)
            (self.x_test, self.y_test) = self.load_test_partition_for_100_clients(idx=client_partition_idx)

        # Partitioned BatchDataset
        self.train_data = tf.data.Dataset.from_tensor_slices({'image': self.x_train, 'label': self.y_train}).batch(32)
        self.val_data = tf.data.Dataset.from_tensor_slices({'image': self.x_test, 'label': self.y_test}).batch(32)

    def load_train_partition_for_10_clients(idx: int):
        assert idx in range(10)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = tf.cast(x_train, dtype=tf.float32)
        # process the same dataset
        return (x_train[idx * 5000 : (idx + 1) * 5000], y_train[idx * 5000 : (idx + 1) * 5000])

    def load_test_partition_for_10_clients(idx : int):
        assert idx in range(10)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = tf.cast(x_test, dtype=tf.float32)
        return (x_test[idx * 1000 : (idx + 1) * 1000], y_test[idx * 1000 : (idx + 1) * 1000])
    
    def load_train_partition_for_100_clients(idx: int):
        # 500/100 train/test split per partition e.g. per client
        # create partition with train/test data per client; note that 600 images per client for 100 clients is convention; 300 images for 200 shards for 2 shards per client is another method and not general convention, but a test
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        assert idx in range(100)
        # 5000/50000 --> 500/50000
        return (x_train[idx * 500: (idx + 1) * 500], y_train[idx * 500: (idx + 1) * 500])

    def load_test_partition_for_100_clients(idx : int):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        assert idx in range(100)
        return (x_test[idx * 100: (idx + 1) * 100], y_test[idx * 100: (idx + 1) * 100])


def build_base_model(params : HParams):
    input_layer = layers.Input(shape=(28, 28, 1), batch_size=None, name="image")
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
    input_layer = layers.Input(shape=(28, 28, 1), batch_size=None, name="image")
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
    input_layer = layers.Input(shape=(28, 28, 1), batch_size=None, name="image")
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
    # precondition matches state check
    input_layer = layers.Input(shape=(28, 28, 1), batch_size=None, name="image")
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
    return adv_model

def setup_client_parse_args():
    parser = argparse.ArgumentParser(description="Crypton Client")
    # configurations
    parser.add_argument("--client_partition_idx", type=int, choices=range(0,10), required=False, default=1)
    parser.add_argument("--adv_grad_norm", type=str, required=False, default="infinity")
    parser.add_argument("--adv_multiplier", type=float, required=False, default=0.2)
    parser.add_argument("--adv_step_size", type=float, choices=range(0, 1), required=False, default=0.05)
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--epochs", type=int, required=False, default=1)
    parser.add_argument("--steps_per_epoch", type=int, required=False, default=0)
    parser.add_argument("--num_clients", type=int, required=False, default=10)
    parser.add_argument("--num_classes", type=int, required=False, default=10)
    parser.add_argument("--nsl_reg", type=bool, required=False, default=True)
    parser.add_argument("--gaussian_reg", type=bool, required=False, default=True)
    # given that there's a set of image corruptions and that 1 can only be applied at a time, should we measure with different types of image corruptions then? Perhaps test the idea of non-convex transformations and their effect on CNN mechanics, or perhaps sets up the discussion to address these situational nuances as a result of the corruption of choice.
    parser.add_argument("--corruption_reg", type=bool, required=False, default=True)
    # todo: add specific corruption attacks for regularization rather than "corruption as regularization" 
    return parser

# config object store resetted after each client execution
client_configs = []
dataset_configs = []

def main(args):
    '''
    Use the `args` parameter to configure the experiment variables. Store configuration objects in their temporary lists to be accessed upon execution outside of the main(args) function.
    
    '''

    params = HParams(num_classes=args.num_classes, adv_multiplier=args.adv_multiplier, adv_step_size=args.adv_step_size, adv_grad_norm=args.adv_grad_norm)
    # configure client train/test partition based on the client partition idx
    dataset_config = DatasetConfig(args.client_partition_idx)
    dataset_configs.append(dataset_config)
    
    # build models
    adv_model = build_adv_model(params=params)
    base_model = build_base_model(params=params)
    gaussian_base_model = build_gaussian_base_model(params=params)
    gaussian_adv_model = build_gaussian_adv_model(params=params)

    # select model
    if (args.adv_reg):
        if (args.gaussian_layer):
            model = gaussian_adv_model
        else:
            model = adv_model

    elif (args.gaussian_layer and args.adv_reg == False):
        model = gaussian_base_model

    else:
        model = base_model
    
    # setup config objects for clients 
    adv_client_config = AdvRegClientConfig(model=model, params=params, train_dataset=dataset_config.train_data, test_dataset=dataset_config.val_data, validation_steps=dataset_config.val_steps)
    client_config = ClientConfig(model=model, params=params, train_dataset=dataset_config.train_data, test_dataset=dataset_config.val_data, validation_steps=dataset_config.val_steps)
    client_configs.append(adv_client_config)
    client_configs.append(client_config)

if __name__ == '__main__':
    # setup client parser to handle user args
    client_parser = setup_client_parse_args()
    args = client_parser.parse_args()
    main(args)
        
    class AdvRegClient(flwr.client.KerasClient):
        def get_weights(self):
            return client_configs[0].model.get_weights()

        def fit(self, parameters, config):
            client_configs[0].model.set_weights(parameters)
            # dataset_config creates the partitions, and loads the partition based on the index (in .sh loop) for the train/val data BatchDataset objects to be passed in the client_configs[0] object, so that the data in each client config object is the partition only, not the original dataset
            history = client_configs[0].model.fit(client_configs[0].train_dataset, validation_data=client_configs[0].test_dataset, validation_steps=dataset_configs[0].val_steps, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs)
            results = {
                "loss": history.history["loss"],
                "sparse_categorical_crossentropy": history.history["sparse_categorical_crossentropy"],
                "sparse_categorical_accuracy": history.history["sparse_categorical_accuracy"],
                "scaled_adversarial_loss": history.history["scaled_adversarial_loss"],
            }

            train_cardinality = len(client_configs[0].train_dataset)
            accuracy = results["sparse_categorical_accuracy"]
            accuracy = int(accuracy[0])
            return client_configs[0].model.get_weights(), train_cardinality, accuracy

        def evaluate(self, parameters, config):
            client_configs[0].model.set_weights(parameters)
            results = client_configs[0].model.evaluate(client_configs[0].test_dataset, verbose=1)
            # only fit uses validation accuracy and sce loss
            results = {
                    "loss": results[0],
                    "sparse_categorical_crossentropy": results[1],
                    "sparse_categorical_accuracy": results[2],
                    "scaled_adversarial_loss": results[3],
            }

            loss = int(results["loss"])
            accuracy = int(results["sparse_categorical_accuracy"])
            test_cardinality = len(client_configs[0].test_dataset)

            return loss, test_cardinality, accuracy

    class Client(flwr.client.KerasClient):
        def get_weights(self):
            return client_configs[1].model.get_weights()

        def fit(self, parameters, config):
            client_configs[1].model.set_weights(parameters)
            history = client_configs[1].model.fit(client_configs[1].train_dataset, validation_data=client_configs[1].test_dataset, validation_steps=dataset_configs[0].val_steps, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs)
            results = {
                "loss": history.history["loss"],
                "sparse_categorical_crossentropy": history.history["sparse_categorical_crossentropy"],
                "sparse_categorical_accuracy": history.history["sparse_categorical_accuracy"],
                "scaled_adversarial_loss": history.history["scaled_adversarial_loss"],
            }

            train_cardinality = len(client_configs[1].train_dataset)
            accuracy = results["sparse_categorical_accuracy"]
            accuracy = int(accuracy[0])

            return client_configs[1].model.get_weights(), train_cardinality, accuracy

        def evaluate(self, parameters, config):
            client_configs[1].model.set_weights(parameters)
            results = client_configs[1].model.evaluate(client_configs[1].test_dataset, verbose=1)
            results = {
                    "loss": results[0],
                    "sparse_categorical_crossentropy": results[1],
                    "sparse_categorical_accuracy": results[2],
                    "scaled_adversarial_loss": results[3],
            }

            loss = int(results["loss"])
            accuracy = int(results["sparse_categorical_accuracy"])
            test_cardinality = len(client_configs[1].test_dataset)

            return loss, test_cardinality, accuracy
    
    flwr.client.start_keras_client(server_address="[::]:8080", client=AdvRegClient())
