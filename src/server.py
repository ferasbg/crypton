import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from client import HParams, build_adv_model

warnings.filterwarnings("ignore")

# create models
params = HParams(num_classes=10, adv_multiplier=0.2, adv_step_size=0.05, adv_grad_norm="infinity")
adv_model = build_adv_model(params=params)

# def main() -> None:
    # setup parse_args with respect to passing relevant params to server.py and client.py instead of run.sh or aggregate file

    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    # 3. since adv. reg. is a client-specific optimization, it's set to False for server-side param. evaluation
    # parameters = HParams(num_classes=10, adv_multiplier=0.2, adv_step_size=0.05, adv_grad_norm="infinity")
    # model = build_base_model(parameters=parameters)

    # # Create strategy
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

    # # Start Flower server for ten rounds of federated learning
    # flwr.server.start_server(strategy=strategy, server_address="[::]:8080", config={"num_rounds": 10})

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    (x_train, y_train) = tf.keras.datasets.mnist.load_data()

    # Use the last 5k training examples as a validation set
    x_test, y_test = x_train[45000:50000], y_train[45000:50000]
    for batch in x_test:
        # with respect to HParams object?
        adv_model.perturb_on_batch(batch)

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
    # add strategy of server-side model (no adv_reg)
    flwr.server.start_server(server_address="[::]:8080", config={"num_rounds": 10})
