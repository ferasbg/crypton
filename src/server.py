import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
import argparse
import flwr
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import (FaultTolerantFedAvg, FedAdagrad, FedAvg,
                                  FedFSv1, Strategy, fault_tolerant_fedavg, fedopt, QffedAvg, FastAndSlow)

import keras
import matplotlib.pyplot as plt
import neural_structured_learning as nsl
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers, optimizers, regularizers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.datasets import cifar10, cifar100
from keras.datasets.cifar10 import load_data
from keras.layers.core import Lambda
from keras.models import Input, Model, Sequential, load_model, save_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.backend import update
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.ops.gen_batch_ops import Batch
from utils import *
from client import *
from numpy import array, float32

from flwr.common import (
    FitRes,
    Parameters,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)

warnings.filterwarnings("ignore")

# acc/loss
server_level_accuracy = []
server_level_loss = []

def build_base_server_model(num_classes : int):
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
    output_layer = layers.Dense(num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')(dense1)
    model = keras.Model(inputs=input_layer, outputs=output_layer, name='client_model')
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model

def main(args) -> None:
    # setup parse_args with respect to passing relevant params to server.py and client.py instead of run.sh or aggregate file

    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    # 3. since adv. reg. is a client-specific optimization, it's set to False for server-side param. evaluation

    # create model
    model = build_base_server_model(num_classes=10)

    if (args.strategy == "fedavg"):
        strategy = FedAvg()

    if (args.strategy == "ft_fedavg"):
        strategy = FaultTolerantFedAvg(fraction_fit=0.3,
            fraction_eval=0.2,
            min_fit_clients=3,
            min_eval_clients=2,
            min_available_clients=10,
            eval_fn=get_eval_fn(model),
            # strategy based on user-written wrapper functions
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
            initial_parameters=model.get_weights())

    if (args.strategy == "fed_adagrad"):
        # this code is untested.
        weights : Weights = model.get_weights()
        strategy = FedAdagrad(
            eta=0.1,
            eta_l=0.316,
            tau=0.5,
            initial_parameters=weights_to_parameters(weights),
        )

    # todo: add remaining strategies (adaptive, non-adaptive, eg FedAdam, FedYogi, FedFSL)
    flwr.server.start_server(strategy=strategy, server_address="[::]:8080", config={"num_rounds": args.num_rounds})

    server_metrics = {
        "server_level_accuracy": server_level_accuracy,
        "server_level_loss": server_level_loss
    }

    server_json_path = './metrics/server_metrics.json'

    with open(server_json_path, 'w') as server_file:
        json.dump(server_metrics, server_file)

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_test, y_test = x_train[45000:50000], y_train[45000:50000]
    val_data = tf.data.Dataset.from_tensor_slices({'image': x_test, 'label': y_test}).batch(batch_size=32)
    params = HParams(num_classes=10, adv_multiplier=0.2, adv_step_size=0.05, adv_grad_norm="infinity") # fixed values
    adv_model = build_adv_model(params=params)

    for batch in val_data:
        adv_model.perturb_on_batch(batch)

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: flwr.common.Parameters,
    ) -> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:

        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_test, y_test)
        server_level_accuracy.append(accuracy)
        server_level_loss.append(loss)

        return loss, {"accuracy": accuracy}

    return evaluate

def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config

def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}

def setup_server_parser():
    parser = argparse.ArgumentParser(description="Crypton Server")
    # configurations
    parser.add_argument("--num_rounds", type=int, required=False, default=3)
    # options = "fedavg", "fedadagrad", "fault_tolerant_fedavg"
    parser.add_argument("--strategy", type=str, required=False, default="fedavg")
    # hardcode or configure with args? optional
    parser.add_argument("--fraction_fit", type=float,
                        required=False, default=0.05)
    parser.add_argument("--fraction_eval", type=float,
                        required=False, default=0.5)
    parser.add_argument("--min_fit_clients", type=int,
                        required=False, default=10)
    parser.add_argument("--min_eval_clients", type=int,
                        required=False, default=2)
    parser.add_argument("--min_available_clients",
                        type=int, required=False, default=2)
    parser.add_argument("--adv_grad_norm", type=str, required=False, default="infinity")
    parser.add_argument("--num_classes", type=int, required=False, default=10)
    parser.add_argument("--adv_multiplier", type=float, required=False, default=0.2)
    parser.add_argument("--adv_step_size", type=float, required=False, default=0.05)
    parser = parser.parse_args()
    return parser

if __name__ == "__main__":
    # get the plot data for the server-side model evaluation (so that we can compare regularized clients to server-model under attack)
    args = setup_server_parser()
    main(args)

