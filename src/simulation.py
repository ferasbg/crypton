import typing
import itertools
from itertools import product, permutations
from client import *
from server import *
from settings import *
import flwr
from flwr.server.strategy import (FaultTolerantFedAvg, FedAdagrad, FedAvg,
                                  FedFSv1, Strategy, fedopt)

# todo: write start_client, start_server, start_simulation
# todo: write setup config based on settings


# configurations are based on the number of strategies, gaussian_noise_layer, image corruptions, nsl adversarial regularization or base regularization (batch, dropout), adversarial hyperparameters, clients, server model config, partitions, etc
client_config = {

}

server_config = {

}

def create_experiment_permutations(client_config : dict, server_config : dict):
    return {}, {}

def start_client():
    pass

def start_server(server_address : str, strategy : Strategy, num_rounds : int, num_clients : int, **kwargs):
    pass

def start_client(model, train_partition, test_partition, **kwargs):
    pass

def start_simulation(num_clients : int, args):
    pass



