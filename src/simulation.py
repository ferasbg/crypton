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
# todo: create exp config permutations given settings
# todo: add configurations eg strategies, gaussian_noise_layer, image corruptions, nsl adversarial regularization or base regularization (batch, dropout), adversarial hyperparameters, clients, server model config, partitions, etc

def create_experiment_permutations(client_config : dict, server_config : dict):
    return {}, {}

# create permutations given dictionary of client_config
client_settings = {
    "adv_reg": False,
    "": "",
    "": "",
    "": "",
    "": "",
    "": "",
    "": "",
    "": "",
    "": "",
    "": "",

}

server_settings = {
    "num_rounds": 10,
    "": "",
    "": "",
    "": "",
    "": "",
    "": "",
    "": "",
    "": "",
    "": "",
    "": "",
}

client_setting_combinations = itertools.combinations(client_settings)