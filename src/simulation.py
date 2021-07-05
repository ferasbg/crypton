from client import *
from server import *
import seaborn as sns
import bokeh as bkh
import chartify
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Process
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from client import *
from server import *
from certify import *

NUM_CLIENTS = 10 
NUM_NORM_TYPES = 2
NORM_TYPES = ["infinity" "l2"] # l2 hasn't been tested
NUM_NORM_VALUES = 10
# currently supporting 1; need FedAdagrad as MVD for strategies
NUM_FEDERATED_STRATEGIES = 1
NUM_ROUNDS = 1 
NUM_EPOCHS = 25 
NUM_STEPS_PER_EPOCH = 1
NORM_RANGE = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
NUM_ADV_REG_TECHNIQUES = 13

'''
    Store metrics, plot data, plot creation function utilities, plot processing (mutator methods), and process callback objects from the experiment.

    Store table config data (to create the table) in terms of pd.DataFrame objects.

    Process: update the static object inside ExperimentConfig inside the client objects and the server creation functions. 

    utils.Plot stores the functions to create each table/plot object, but the Experimentconfig object stores the actual data to be passed.  

    Variables will be to configure the plot required, and to store the lists that are required to create the plots (x-y coordinates --> two lists).

    Storing variables in object guarantee flushed data and flexibility with process when creating plots.

'''
# need to track the train losses/accuracy, robust accuracy (we can match the index with the adv_step_size intensity eg with the list of norm values in NORM_RANGE)
communication_rounds = 10
# you do this iteratively: experiment_config.client_federated_train_adversarial_loss_set.append(results["scaled_adversarial_loss"])
client_federated_train_pristine_loss_set = []
client_federated_train_adversarial_loss_set = []
client_federated_evaluation_adversarial_loss_set = []
client_federated_evaluation_pristine_loss_set = []
adversarial_server_federated_evaluation_loss_set = []
client_federated_train_accuracy_set = []
client_federated_eval_accuracy_set = []
client_federated_robust_train_accuracy_set = []
server_side_federated_robust_accuracy_set = []

# adversarial regularization techniques
adversarial_regularization_techniques_set = ["neural_structured_learning", "gaussian_noise_regularization", "data_corruption_regularization", "blur_corruption_regularization", "noise_corruption_regularization"]
blur_corruptions_set = ["motion_blur", "glass_blur", "zoom_blur", "gaussian_blur", "defocus_blur"]
data_corruption_set = ["jpeg_compression", "elastic_transform", "pixelate"]
noise_corruption_set = ["shot_noise", "impulse_noise", "speckle_noise"]

def setup_simulation_parser():
    parser = argparse.ArgumentParser(description="Crypton Single-Machine Simulation System.")
    parser.add_argument("--client_partition_idx", type=int, required=False, default=0)
    parser.add_argument("--adv_grad_norm", type=str, required=False, default="infinity")
    parser.add_argument("--adv_multiplier", type=float, required=False, default=0.2)
    parser.add_argument("--adv_step_size", type=float, required=False, default=0.05)
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--epochs", type=int, required=False, default=1)
    parser.add_argument("--steps_per_epoch", type=int, required=False, default=1)
    parser.add_argument("--num_clients", type=int, required=False, default=10)
    parser.add_argument("--num_classes", type=int, required=False, default=10)
    parser.add_argument("--model", type=str, required=False, default="nsl_model")
    parser.add_argument("--client", type=str, required=False, default="nsl_client")
    parser.add_argument("--nsl_reg", type=bool, required=False, default=False)
    parser.add_argument("--gaussian_reg", type=bool, required=False, default=False)
    parser.add_argument("--corruption_name", type=str, required=False, default="")
    parser.add_argument("--nominal_reg", type=str, required=False, default=False)
    parser.add_argument("--num_rounds", type=int, required=False, default=3)
    parser.add_argument("--strategy", type=str, required=False, default="fedavg")
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
    parser = parser.parse_args()
    return parser

def start_server(num_clients : int, fraction_fit : float, args):
    model = build_base_server_model(num_classes=10)
    
    ft_fed_avg = FaultTolerantFedAvg(fraction_fit=fraction_fit,
        fraction_eval=0.2,
        min_fit_clients=3,
        min_eval_clients=2,
        min_available_clients=num_clients,
        eval_fn=get_eval_fn(model),
        # strategy based on user-written wrapper functions
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=model.get_weights())

    fed_avg = FedAvg(
        fraction_fit=fraction_fit,
        fraction_eval=0.2,
        min_fit_clients=3,
        min_eval_clients=2,
        min_available_clients=num_clients,
        eval_fn=get_eval_fn(model),
        # strategy based on user-written wrapper functions
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=model.get_weights()
    )

    # fed_adagrad = FedAdagrad(initial_parameters=tf.convert_to_tensor(value=model.get_weights()))

    if (args.strategy == "fedavg"):
        strategy = fed_avg 

    if (args.strategy == "ft_fedavg"):
        strategy = ft_fed_avg

    if (args.strategy == "fed_adagrad"):
        strategy = None

    # hardcode
    strategy = fed_avg

    flwr.server.start_server(strategy=strategy, server_address="[::]:8080", config={"num_rounds": args.num_rounds})
    
def start_nsl_client(client_train_partition_dataset, client_test_partition_dataset, dataset_config : DatasetConfig, model : AdversarialRegularization, args):

    class AdvRegClient(flwr.client.KerasClient):
        def get_weights(self):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            history = model.fit(client_train_partition_dataset, validation_data=client_test_partition_dataset, validation_steps=dataset_config.partitioned_val_steps, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs)
            results = {
                "loss": history.history["loss"],
                "sparse_categorical_crossentropy": history.history["sparse_categorical_crossentropy"],
                "sparse_categorical_accuracy": history.history["sparse_categorical_accuracy"],
                "scaled_adversarial_loss": history.history["scaled_adversarial_loss"],
            }
            client_federated_adversarial_loss = results["scaled_adversarial_loss"]
            client_federated_train_adversarial_loss_set.append(client_federated_adversarial_loss)

            train_cardinality = len(client_train_partition_dataset)
            accuracy = results["sparse_categorical_accuracy"]
            accuracy = int(accuracy[0])
            client_federated_train_accuracy_set.append(accuracy)

            return model.get_weights(), train_cardinality, accuracy

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            results = model.evaluate(client_test_partition_dataset, verbose=1)
            # only fit uses validation accuracy and sce loss; take data point of loss for each client for each round (not epoch)
            results = {
                    "loss": results[0],
                    "sparse_categorical_crossentropy": results[1],
                    "sparse_categorical_accuracy": results[2],
                    "scaled_adversarial_loss": results[3],
            }

            loss = int(results["loss"])
            accuracy = int(results["sparse_categorical_accuracy"])
            # client_configs[0].test_dataset
            test_cardinality = len(client_test_partition_dataset)

            return loss, test_cardinality, accuracy
    
    client = AdvRegClient()
    flwr.client.start_keras_client(server_address="[::]:8080", client=client)

def start_client(client_train_partition_dataset, client_test_partition_dataset, dataset_config : DatasetConfig, model : tf.keras.models.Model, args):
    class Client(flwr.client.KerasClient):
        def get_weights(self):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            history = model.fit(client_train_partition_dataset, validation_data=client_test_partition_dataset, validation_steps=dataset_config.partitioned_val_steps, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs)
            results = {
                "loss": history.history["loss"],
                "sparse_categorical_crossentropy": history.history["sparse_categorical_crossentropy"],
                "sparse_categorical_accuracy": history.history["sparse_categorical_accuracy"],
                "scaled_adversarial_loss": history.history["scaled_adversarial_loss"],
            }

            train_cardinality = len(client_train_partition_dataset)
            accuracy = results["sparse_categorical_accuracy"]
            accuracy = int(accuracy[0])

            return model.get_weights(), train_cardinality, accuracy

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            results = model.evaluate(client_test_partition_dataset, verbose=1)
            results = {
                    "loss": results[0],
                    "sparse_categorical_crossentropy": results[1],
                    "sparse_categorical_accuracy": results[2],
                    "scaled_adversarial_loss": results[3],
            }

            loss = int(results["loss"])
            accuracy = int(results["sparse_categorical_accuracy"])
            test_cardinality = len(client_test_partition_dataset)

            return loss, test_cardinality, accuracy
    
    client = Client()
    flwr.client.start_keras_client(server_address="[::]:8080", client=client)

def run_simulation(num_rounds: int, num_clients: int, fraction_fit: float, args):
    """
    Execute each experimental configuration.

    @param args : ArgumentParser will store the configurations for the current exp_config. This will be executed for each experimental configuration permutation.

    """

    # This will hold all the processes which we are going to create
    processes = []

    # specify 
    server_process = Process(
        target=start_server, args=(num_rounds, fraction_fit, args)
    )

    # .start() implies instantiation so that when you iterate over this list it runs these functions iteratively without a multiplexer 
    server_process.start()
    processes.append(server_process)

    # 60 seconds to apply perturbation attack to server-side evaluation data
    time.sleep(60)
    dataset_config = DatasetConfig(args)
    client_train_partitions = []
    client_test_partitions = []

    for i in range(args.num_clients):
        # load partitions
        client_train_partition = dataset_config.load_train_partition(idx=i)
        client_test_partition = dataset_config.load_test_partition(idx=i)
        
        # append partitions to list to iterate from
        client_train_partitions.append(client_train_partition)
        client_test_partitions.append(client_test_partition)
    
    assert len(client_train_partitions) == args.num_clients
    assert len(client_test_partitions) == args.num_clients

    # Start all the clients
    for i in range(args.num_clients):
        params = HParams(num_classes=args.num_classes, adv_multiplier=args.adv_multiplier, adv_step_size=args.adv_step_size, adv_grad_norm=args.adv_grad_norm)

        if (args.model == "nsl_model" or args.nsl_reg == True):
            model = build_adv_model(params=params)
            client_process = Process(target=start_nsl_client, args=(client_train_partitions[i], client_test_partitions[i], dataset_config, model, args))
            client_process.start()
            processes.append(client_process)
        
        elif (args.model == "gaussian_model"):
            model = build_gaussian_base_model(params=params)
            client_process = Process(target=start_client, args=(client_train_partitions[i], client_test_partitions[i], dataset_config, model, args))
            client_process.start()
            processes.append(client_process)
        
        elif (args.model == "base_model"):
            model = build_base_model(params=params)
            client_process = Process(target=start_client, args=(client_train_partitions[i], client_test_partitions[i], dataset_config, model, args))
            client_process.start()
            processes.append(client_process)
            
    # Block until all processes are finished
    for p in processes:
        p.join()

def main(args) -> None:
    # args stores the args for the exp_config for all exp_configs generated by the nested loop that forms ordered permutations
    run_simulation(num_rounds=args.num_rounds, num_clients=args.num_clients, fraction_fit=0.5, args=args)

# plan: based on target variables to update, pass them to args and then simulation will run iterative process that can map to the end result of the plots needed; iterate for each plot and define the commands to run in .sh file
if __name__ == '__main__':
    simulation_args = setup_simulation_parser()
    main(simulation_args)