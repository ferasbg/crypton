from prediction.network import Network
from crypto.mpc_net import MPCNet
from specification_auth import Specification_Auth

class Deploy(Network, MPCNet):
    def __init__(self):
        self.network = Network()
        self.mpc_network = MPCNet()
        self.specification_auth = Specification_Auth()


    def main():
        """Runner Method for Sequential Computation of Server Nodes """
        pass


if __name__ == '__main__':
    Deploy()
    Deploy.main()

