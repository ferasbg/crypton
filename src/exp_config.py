from simulation import *
from client import *
from server import *

exp_config = {
    "plot_1": {
        "x_axis": "Communication Rounds",
        "y_axis": "Server Test Accuracy",
        "x_coordinates": {
            "num_rounds": 1,
            "num_rounds": 2,
            "num_rounds": 3,
            "num_rounds": 4,
            "num_rounds": 5,
            "num_rounds": 6,
            "num_rounds": 7,
            "num_rounds": 8,
            "num_rounds": 9,
            "num_rounds": 10,
        },
        
        "y_coordinates": {
            # access from the config object that stores the server's evaluation accuracy from flwr log, eval metrics, History object
            "server_test_accuracy": 1,
            "server_test_accuracy": 2,
            "server_test_accuracy": 3,
            "server_test_accuracy": 4,
            "server_test_accuracy": 5,
            "server_test_accuracy": 6,
            "server_test_accuracy": 7,
            "server_test_accuracy": 8,
            "server_test_accuracy": 9,
            "server_test_accuracy": 10,
        },
        "norm_type": ["infinity", "l2"],
        "line_names": {
            # these lines are created based on the target variables you want to measure for and at what scope
            "line_1": "FedAvg" + " ",
            "line_2": "FedAdagrad" + " ",
            "line_2": "FedAdagrad" + " ",

        }
    }
}

exp_config_schema_set = []