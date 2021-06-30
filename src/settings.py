import flwr
from flwr.server.strategy import FedAdagrad, FedAvg, FedFSv0, FedFSv1, FaultTolerantFedAvg
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, weights_to_parameters
from flwr.server.client_proxy import ClientProxy
from server import *
from client import AdvRegClient, Client
from adversarial import HParams, build_adv_model, build_base_model
import argparse
import absl
from absl import flags
import tensorflow_datasets as tfds
from dataset import *

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
    - pseudorandom_image_distribution_transformation_train : bool (default=False, options=[False, False])
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
DEFAULT_SERVER_ADDRESS="[::]:8080"
DEFAULT_NUM_CLIENTS = 10
DEFAULT_NUM_ROUNDS = 10
DEFAULT_CLIENT = Client()
ADV_REG_CLIENT = AdvRegClient()
CLIENT_SET = [DEFAULT_CLIENT, ADV_REG_CLIENT]
DEFAULT_ADV_GRAD_NORM = "infinity"
ADV_GRAD_NORM_OPTIONS = ["infinity", "l2"]
DEFAULT_ADV_MULTIPLIER = 0.2
DEFAULT_ADV_STEP_SIZE = 0.05
DEFAULT_CLIENT_LR_SCHEDULE = []
DEFAULT_SERVER_LR_SCHEDULE = []
CLIENT_LEARNING_RATE_SCHEDULER = tf.keras.callbacks.LearningRateScheduler(schedule=DEFAULT_CLIENT_LR_SCHEDULE)
SERVER_LEARNING_RATE_SCHEDULER = tf.keras.callbacks.LearningRateScheduler(schedule=DEFAULT_SERVER_LR_SCHEDULE)
PARAMETERS = HParams(num_classes=DEFAULT_NUM_CLIENTS, adv_multiplier=DEFAULT_ADV_MULTIPLIER, adv_step_size=DEFAULT_ADV_STEP_SIZE, adv_grad_norm=ADV_GRAD_NORM_OPTIONS[0])
ADVERSARIAL_REGULARIZED_MODEL = build_adv_model(parameters=PARAMETERS)
BASE_MODEL = build_base_model(parameters=PARAMETERS)
DEFAULT_MODEL = BASE_MODEL
DEFAULT_FRACTION_FIT = 0.3
DEFAULT_FRACTION_EVAL = 0.2
DEFAULT_MIN_FIT_CLIENTS = 2
DEFAULT_MIN_EVAL_CLIENTS = 2
DEFAULT_MIN_AVAILABLE_CLIENTS = 10
DEFAULT_INITIAL_SERVER_MODEL_PARAMETERS = weights_to_parameters(DEFAULT_MODEL.get_weights())

federated_averaging = flwr.server.strategy.FedAvg(
            fraction_fit=DEFAULT_FRACTION_FIT,
            fraction_eval=DEFAULT_FRACTION_EVAL,
            min_fit_clients=DEFAULT_MIN_FIT_CLIENTS,
            min_eval_clients=DEFAULT_MIN_EVAL_CLIENTS,
            min_available_clients=DEFAULT_MIN_AVAILABLE_CLIENTS,
            eval_fn=get_eval_fn(DEFAULT_MODEL),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
            initial_parameters=weights_to_parameters(DEFAULT_MODEL.get_weights()),
    )

federated_adaptive_optimization = FedAdagrad(
        fraction_fit=0.3,
        fraction_eval=0.2,
        min_fit_clients=101,
        min_eval_clients=101,
        min_available_clients=110,
        eval_fn=get_eval_fn(DEFAULT_MODEL),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        accept_failures=False,
        initial_parameters=weights_to_parameters(DEFAULT_MODEL.get_weights()),
        tau=1e-9,
        eta=1e-1,
        eta_l=1e-1
)



DEFAULT_FEDERATED_STRATEGY_SET = [federated_averaging, federated_adaptive_optimization]
DEFAULT_TRAIN_EPOCHS = 5
DEFAULT_CLIENT_LEARNING_RATE = 0.1
DEFAULT_SERVER_LEARNING_RATE = 0.1
# DEFAULT_WEIGHT_REGULARIZATION = add in DEFAULT MODEL layer
DEFAULT_GAUSSIAN_STATE = False
DEFAULT_IMAGE_CORRUPTION_STATE = False
# based on the set I want to apply that specific image corruption
MISC_CORRUPTION_SET = ["spatter", "saturate", "fog", "brightness", "contrast"]
BLUR_CORRUPTION_SET = ["motion_blur", "glass_blur", "zoom_blur", "gaussian_blur", "defocus_blur"]
DATA_CORRUPTION_SET = ["jpeg_compression", "elastic_transform", "pixelate"]
NOISE_CORRUPTION_SET = ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"]
# this defines whether a particular set of corruptions are applied
TRAIN_SET_IMAGE_DISTORTION_STATE = False
SERVER_TEST_SET_PERTURBATION_STATE = False
