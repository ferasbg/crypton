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
from settings import *
from client import *
from server import *
from certify import *

'''
Exp Config Algorithm:
    # best form of execution is to loop over the independent variable set (federated strategy, adv reg techniques, norm types, norm values eg radius eg epsilon value)
    # add the data to a list and then pass it as a parameter to the plot functions that will graph the situation in terms of all the specified graphs
    # we want to iterate correctly in terms of the exp config then run the server.py file and the client.py file in terms of the partitions but making sure that we iterate over every client (so run it between range 0-9)

'''

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

# Todo: Use https://github.com/locuslab/smoothing/tree/master/analysis/plots in order to create the plots. 
# todo: setup plots below per exp config in process
# todo: write the formalizations (on paper, and checks defined in certify) of the problem (evaluating adversarial robustness without tight certifications) is manage-able; albeit certification isn't to the degree of standard robustness certifications (eg. randomized smoothing, etc)
# todo: still not clear on how the graphs will look like; need to do that before translating to code
# todo: scale to 100 rounds with 100 clients with utils.Data and server args
# todo: define the graphs per exp config iteration and save the plots
# todo: store metrics in pandas.Dataframe per exp config, then append the exp configs based on the table you defined earlier
# todo: create graphs out of the existing table, fit it to features for comparison against baselines (eg adv reg and strategy permutations, etc --> other variables)
# todo: define exp_config iteration in main

# need to track the train losses/accuracy, robust accuracy (we can match the index with the adv_step_size intensity eg with the list of norm values in NORM_RANGE)
communication_rounds = [0, 2, 4, 6, 8, 10]

client_federated_train_pristine_loss_set = []
client_federated_train_adversarial_loss_set = []
client_federated_evaluation_adversarial_loss_set = []
client_federated_evaluation_pristine_loss_set = []
adversarial_server_federated_evaluation_loss_set = []

client_federated_train_accuracy_set = []
client_federated_eval_accuracy_set = []
client_federated_robust_train_accuracy_set = []
server_side_federated_robust_accuracy_set = []

corruption_types_set = ["blur", "noise", "data"]
'''
Hendrycks, Dan and Dietterich, Thomas G.
Benchmarking Neural Network Robustness to Common Corruptions and
Surface Variations

'''
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
    parser.add_argument("--steps_per_epoch", type=int, required=False, default=0)
    parser.add_argument("--num_clients", type=int, required=False, default=10)
    parser.add_argument("--num_classes", type=int, required=False, default=10)
    parser.add_argument("--model", type=str, required=False, default="nsl_model")
    parser.add_argument("--nsl_reg", type=bool, required=False, default=False)
    parser.add_argument("--gaussian_reg", type=bool, required=False, default=False)
    parser.add_argument("--nominal_reg", type=str, required=False, default=True)
    parser.add_argument("--corruption_name", type=str, required=False, default="")
    parser.add_argument("--client", type=str, required=False, default="nsl_client")
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
    strategy = FedAvg(min_available_clients=num_clients, fraction_fit=fraction_fit)
    flwr.server.start_server(strategy=strategy, server_address=DEFAULT_SERVER_ADDRESS, config={"num_rounds": args.num_rounds})
    
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
    flwr.client.start_keras_client(server_address=DEFAULT_SERVER_ADDRESS, client=client)

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
    flwr.client.start_keras_client(server_address=DEFAULT_SERVER_ADDRESS, client=client)

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

    # Start all the clients
    for i in range(args.num_clients):
        params = HParams(num_classes=args.num_classes, adv_step_size=args.adv_step_size, adv_grad_norm=args.adv_grad_norm)

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

# define EVERY value that is used in the simulation_parser and configure these variables as params
def create_args_parser(client_partition_idx : int, adv_grad_norm : str, adv_multiplier=0.2, adv_step_size=0.05, batch_size=32, epochs=25, steps_per_epoch=None, num_clients=10, num_classes=10, model=None, nsl_reg : bool = False, gaussian_reg : bool = False, nominal_reg=True, corruption_name : str = "", client : str = "client", num_rounds=10, strategy="fedavg", fraction_fit=0.5, fraction_eval=0.2, min_fit_clients=2, min_eval_clients=10, dataset_config : DatasetConfig(args=None) = None):
    # hardcode defaults given parameters.
    parser = argparse.ArgumentParser(description="Crypton Exp Config Object.")
    parser.add_argument("--client_partition_idx", type=int, required=False, default=client_partition_idx)
    parser.add_argument("--adv_grad_norm", type=str, required=False, default=adv_grad_norm)
    parser.add_argument("--adv_multiplier", type=float, required=False, default=adv_multiplier)
    parser.add_argument("--adv_step_size", type=float, required=False, default=adv_step_size)
    parser.add_argument("--batch_size", type=int, required=False, default=batch_size)
    parser.add_argument("--epochs", type=int, required=False, default=epochs)
    parser.add_argument("--steps_per_epoch", type=int, required=False, default=steps_per_epoch)
    parser.add_argument("--num_clients", type=int, required=False, default=num_clients)
    parser.add_argument("--num_classes", type=int, required=False, default=num_classes)
    parser.add_argument("--model", type=str, required=False, default=model)
    parser.add_argument("--nsl_reg", type=bool, required=False, default=nsl_reg)
    parser.add_argument("--gaussian_reg", type=bool, required=False, default=gaussian_reg)
    parser.add_argument("--nominal_reg", type=str, required=False, default=nominal_reg)
    parser.add_argument("--corruption_name", type=str, required=False, default=corruption_name)
    # options: "nsl_client", "client"
    parser.add_argument("--client", type=str, required=False, default=client)
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

if __name__ == '__main__':
    client_args = setup_client_parser()
    # iterate over exp_config_set, thus 780 operations of the loop.
    # execute each exp config; end result: plots in /figures directory and all exp configs check out
        # ex: fedavg --> nsl --> l-inf --> l-inf e=0.05 (p-E) --> run_simulation()
    
    for j in range(NUM_FEDERATED_STRATEGIES):
        for i in range(NUM_ADV_REG_TECHNIQUES):
            for n in range(len(ADV_GRAD_NORM_OPTIONS)):
                for s in range(NUM_NORM_VALUES):
                    # iterate in terms of these target variables to get each exp config
                    args = create_args_parser(client_partition_idx=i, adv_grad_norm=ADV_GRAD_NORM_OPTIONS[n], adv_multiplier=0.2, adv_step_size=NUM_NORM_VALUES[s], batch_size=32, epochs=1, steps_per_epoch=0, num_clients= 10, num_classes= 10, model = "base_model", nsl_reg= False, gaussian_reg  = False, nominal_reg  = True, client : str = "client", num_rounds : int = 1, strategy : str = NUM_FEDERATED_STRATEGIES[j], fraction_fit=0.5, fraction_eval=0.2, min_fit_clients=2, min_eval_clients=10, args=client_args, dataset_config=DatasetConfig(client_args))
                    main(args)

    args = setup_simulation_parser()
    main(args)
    