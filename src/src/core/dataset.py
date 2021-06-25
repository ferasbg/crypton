import flwr as fl
from client import Client
import argparse

def main(args):
    fl.start_numpy_client("[::]:8080", client=Client())
    # each config is accessed with the args object param

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crypton")
    parser.add_argument(
        "--num-clients",
        default=10,
        type=int,
    )
    
    parser.add_argument(
        "--local-epochs",
        default=2,
        type=int,
    )
    parser.add_argument("--batch-size", default=32, type=int, help="batch_size")
    parser.add_argument(
        "--learning-rate", default=0.15, type=float, help="learning rate. Modify given learning rate schedule. Check for client/server relations with lr schedule changes."
    )

    args = parser.parse_args()
    main(args)