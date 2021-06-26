import argparse
import argparse
import flwr
from flwr.server.strategy import (FaultTolerantFedAvg, FedAdagrad, FedAvg,
                                  FedFSv1, Strategy, fedopt)

# define experimental configs here, then configure adv_reg model and base model, as well as CifarClient to fit to the configurations defined
# write logic that processes config data from get_parse_args

'''

Configurations:
    - dataset to use (dataset="cifar10" or dataset="cifar100")
    - adv_grad_norm : str (options: adv_grad_norm="infinity"; adv_grad_norm="l2")
    - adv_step_size : float (options: range(0.1, 0.9) --> range differs/depends on the adv_grad_norm argument)
    - batch_size : int (default=32)
    - epochs : int (default=5)
    - steps_per_epoch : int (default=None)
    - num_clients : int
    - num_partitions : int (default=num_clients)
    - num_rounds : int (default=10)
    - federated_optimizer_strategy : str (options: federated_optimizer="fedavg", federated_optimizer="fedadagrad", federated_optimizer="faulttolerantfedavg", federated_optimizer="fedsv1", federated_optimizer="fedopt")
    - adv_reg : bool (default=False)
    - gaussian_layer : bool (default=False)
    - pseudorandom_image_distribution_transformation_train : bool (default=False, options=[True, False])
    - apply_all_image_degradation_configs : bool (default=False, options=[True, False])
    - image_corruption_train : bool
    - image_resolution_loss_train : bool
    - formal_robustness_analysis : bool
    - input_shape = [32, 32, 3]
    - num_classes = num_classes
    - conv_filters = [32, 64, 64, 128, 128, 256]
    - kernel_size = (3, 3)
    - pool_size = (2, 2)
    - num_fc_units = [64]
    - batch_size = 32
    - epochs = 5
    - adv_multiplier = adv_multiplier
    - adv_step_size = adv_step_size
    - adv_grad_norm = adv_grad_norm
    - fraction_fit: float = 0.1,
    - fraction_eval: float = 0.1,
    - min_fit_clients: int = 2,
    - min_eval_clients: int = 2,
    - min_available_clients: int = 2,
    - eval_fn: Optional[
        Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
    ] = None,
    - on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
    - on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
    - accept_failures: bool = True,
    - initial_parameters: Optional[Parameters] = None

'''

# pass parse args in simulation.py
def get_parse_args():
    parser = argparse.ArgumentParser(description="Crypton")

