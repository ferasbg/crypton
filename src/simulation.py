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

class StrategyConfig(object):
    pass

def main(args) -> None:
    # pass map dataset of train dataset only
    train_partitions = Data.create_train_partitions(dataset=[], num_clients=10)
    test_partitions = Data.create_test_partitions(dataset=[], num_clients=10)
    strategy_config = StrategyConfig()  
    params = HParams(10, 0.2, 0.05, "infinity")
    strategy = FedAdagrad()

    # run a process
    for client in range(params.num_clients):
        adv_client_config = AdvRegClientConfig(model=[], params=[], train_dataset=[], test_dataset=[], validation_steps=[])
        client = AdvRegClient()
        flwr.server.start_server(server_address=DEFAULT_SERVER_ADDRESS, strategy=strategy)
        flwr.client.start_keras_client(server_address=DEFAULT_SERVER_ADDRESS, client=client)