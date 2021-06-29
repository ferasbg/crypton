import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
import argparse
import flwr
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import (FaultTolerantFedAvg, FedAdagrad, FedAvg,
                                  FedFSv1, Strategy, fedopt)
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

warnings.filterwarnings("ignore")

def build_base_model(num_classes : int):
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
    model = build_base_model(num_classes=10)

    # # create strategy
    # strategy = flwr.server.strategy.FedAvg(
    #     fraction_fit=0.3,
    #     fraction_eval=0.2,
    #     min_fit_clients=3,
    #     min_eval_clients=2,
    #     min_available_clients=10,
    #     eval_fn=get_eval_fn(model),
    #     # strategy based on user-written wrapper functions
    #     on_fit_config_fn=fit_config,
    #     on_evaluate_config_fn=evaluate_config,
    #     initial_parameters=model.get_weights(),
    # )

    # Start Flower server for ten rounds of federated learning
    flwr.server.start_server(server_address="[::]:8080", config={"num_rounds": 10})

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    (x_train, y_train) = tf.keras.datasets.mnist.load_data()

    # Use the last 5k training examples as a validation set
    x_test, y_test = x_train[45000:50000], y_train[45000:50000]
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    # for batch in train_dataset_for_base_model:
        #     adv_model.perturb_on_batch(batch)
        
        # for batch in test_dataset_for_base_model:
        #     adv_model.perturb_on_batch(batch)

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: flwr.common.Parameters,
    ) -> Optional[Tuple[float, Dict[str, flwr.common.Scalar]]]:

        model.set_weights(weights)  # Update model with the latest parameters
        # convert from tuples to dicts if this
        loss, accuracy = model.evaluate(x_test, y_test)
        # get dict of history in evaluation, and return
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypton Server")
    # configurations
    parser.add_argument("--num_rounds", type=int, required=False)
    parser.add_argument("--federated_optimizer_strategy", type=str, required=False)
    parser.add_argument("--fraction_fit", type=float, required=False, default=0.05)
    parser.add_argument("--fraction_eval", type=float, required=False, default=0.5)
    parser.add_argument("--min_fit_clients", type=int, required=False, default=10)
    parser.add_argument("--min_eval_clients", type=int, required=False, default=2)
    parser.add_argument("--min_available_clients", type=int, required=False, default=2)
    parser.add_argument("--accept_client_failures_fault_tolerance", type=bool, required=False, default=False)
    
    args = parser.parse_args()
    main(args)