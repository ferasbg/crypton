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

# todo: implement formal robustness property checks/formulations in formal_robustness.py; review the paper again and also the specifications paper
# todo : add weight reg in each model function
# todo: add validation set to the .fit() function for both clients
# todo (very imp.): write load_partition set and create_partition set in utils; create client train/test partitions here; for each client instance run with python3 client.py --argName1 etc we will hardcode the configs in a .sh file and run them
# todo: create (train and/or test) partitions functions are run by default which creates the set of train/test partitions respectively, and are accessed based on the current client_idx
# todo: clear up technical process relating to args and config objects that handle the data passed from args
# todo: validate and test base client (control) to make sure that it can be sampled just like AdvRegClient
# todo: (precondition: created client train and test partitions in DatasetConfig and loaded partition into ExperimentConfig and into AdvRegClientConfig): create client partition given index in args.client_partition : int --> will create train/test partitions processed with ExperimentConfig 
# todo: given gaussian_state, pass the specified model into the ClientConfig objects (adv. reg, non adv. reg)

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

class ServerConfig(object):
    # server-side model and server configurations
    def __init__(self):
        self.fed_adagrad = FedAdagrad()
        self.fed_avg = FedAvg()

def normalize(features):
  features['image'] = tf.cast(
      features['image'], dtype=tf.float32) / 255.0
  return features

def convert_to_tuples(features):
  return features['image'], features['label']

def convert_to_dictionaries(image, label, IMAGE_INPUT_NAME='image', LABEL_INPUT_NAME='label'):
  return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}

class DatasetConfig:
    def __init__(self):
        # MapDataset for partition creation
        self.datasets = tfds.load('mnist')
        self.map_train_dataset = self.datasets['train']
        self.map_test_dataset = self.datasets['test']
        self.train_dataset_for_base_model = self.map_train_dataset.map(normalize).shuffle(10000).batch(32).map(convert_to_tuples)
        self.test_dataset_for_base_model = self.map_test_dataset.map(normalize).batch(32).map(convert_to_tuples)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test, y_test = x_train[-10000:], y_train[-10000:]

        self.x_train = tf.cast(x_train, dtype=tf.float32)
        self.y_train = y_train
        self.x_test = tf.cast(x_test, dtype=tf.float32)
        self.y_test = y_test
        self.val_steps = self.x_test.shape[0] / 32

        # type: BatchDataset
        self.train_data = tf.data.Dataset.from_tensor_slices({'image': x_train, 'label': y_train}).batch(32)
        self.val_data = tf.data.Dataset.from_tensor_slices({'image': x_test, 'label': y_test}).batch(32)

        # client partitions created
        self.adv_train_partitions = Data.create_train_partitions(self.x_train, self.y_train, self.x_test, self.y_test, num_clients=10)
        self.adv_test_partitions = Data.create_test_partitions(self.x_train, self.y_train, self.x_test, self.y_test, num_clients=10)
        self.client_train_partitions = Data.create_train_partitions(self.x_train, self.y_train, self.x_test, self.y_test, num_clients=10)
        self.client_test_partitions = Data.create_test_partitions(self.x_train, self.y_train, self.x_test, self.y_test, num_clients=10)

class ExperimentConfig(object):
    '''
    DatasetConfig --> ExperimentConfig --> AdvRegClientConfig --> AdvRegClient
    '''
    def __init__(self, client_config, args, client_partition : int, dataset_config=DatasetConfig()):
        self.client_config = client_config
        self.args = args
        self.client_partition = client_partition
        # pass tuples from (x_train, y_train), (x_test, y_test) and convert into partitioned BatchDataset objects when passing to AdvRegClientConfig object
        self.client_train_partition_dataset = Data.load_train_partition(client_partition, dataset_config.x_train, dataset_config.y_train)
        self.client_test_partition_dataset = Data.load_test_partition(client_partition, dataset_config.x_test, dataset_config.y_test)

class Args:
    # args = Args(args=args).args; only use of this approach is persistent state regardless of code order for params; configuring params with args; agnostic to where param code is written
    def __init__(self, args):
        self.args = args

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

def build_gaussian_base_model():
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

def build_gaussian_adv_model():
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
    parser.add_argument("--num_partitions", type=int, choices=range(0, 10), required=False)
    parser.add_argument("--client_partition", type=int, required=False, default=0)
    parser.add_argument("--adv_grad_norm", type=str, required=False)
    parser.add_argument("--adv_multiplier", type=float, required=False, default=0.2)
    parser.add_argument("--adv_step_size", type=float, choices=range(0, 1), required=False, default=0.05)
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--epochs", type=int, required=False, default=10)
    parser.add_argument("--num_clients", type=int, required=False, default=10)
    parser.add_argument("--adv_reg", type=bool, required=False, default=True)
    parser.add_argument("--gaussian_layer", type=bool, required=False, default=False)
    # nominal regularization shouldn't be configured at all
    parser.add_argument("--weight_regularization", type=bool, required=False, default=True)
    # FedAdagrad assumes Adam(lr=0.01) as the base optimizer on the client-level
    parser.add_argument("--sgd_momentum", type=float, required=False, default=0.9)
    return parser

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# setup models; configure so that it can be setup with args; we could create an args object
params = HParams(num_classes=10, adv_multiplier=0.2, adv_step_size=0.05, adv_grad_norm="infinity")
dataset_config = DatasetConfig()

# setup client configurations; hardcoded configs for now
adv_client_config = AdvRegClientConfig(model=build_adv_model(params=params), params=params, train_dataset=dataset_config.train_data, test_dataset=dataset_config.val_data, validation_steps=dataset_config.val_steps)
client_config = ClientConfig(model=build_base_model(params=params), params=params, train_dataset=dataset_config.train_data, test_dataset=dataset_config.val_data, validation_steps=dataset_config.val_steps)
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

class AdvRegClient(flwr.client.KerasClient):
    def get_weights(self):
        return adv_client_config.model.get_weights()

    def fit(self, parameters, config):
        adv_client_config.model.set_weights(parameters)
        # dataset_config creates the partitions, and loads the partition based on the index (in .sh loop) for the train/val data BatchDataset objects to be passed in the adv_client_config object, so that the data in each client config object is the partition only, not the original dataset
        history = adv_client_config.model.fit(adv_client_config.train_dataset, validation_data=adv_client_config.val_data, validation_steps=dataset_config.val_steps, steps_per_epoch=1, epochs=1)
        results = {
            "loss": history.history["loss"],
            "sparse_categorical_crossentropy": history.history["sparse_categorical_crossentropy"],
            "sparse_categorical_accuracy": history.history["sparse_categorical_accuracy"],
            "scaled_adversarial_loss": history.history["scaled_adversarial_loss"],
        }

        train_cardinality = len(adv_client_config.train_dataset)
        accuracy = results["sparse_categorical_accuracy"]
        accuracy = int(accuracy[0])
        return adv_client_config.model.get_weights(), train_cardinality, accuracy

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

        loss = int(results["loss"])
        accuracy = int(results["sparse_categorical_accuracy"])
        test_cardinality = len(adv_client_config.test_dataset)

        return loss, test_cardinality, accuracy

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
    flwr.client.start_keras_client(server_address="[::]:8080", client=AdvRegClient())

if __name__ == '__main__':
    # client_parser = setup_client_parse_args()
    # args store client partition index
    # args = client_parser.parse_args()
    # experiment_config  = ExperimentConfig(client_config=adv_client_config, args=args, client_partition=0)
    main()
