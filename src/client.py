import argparse
import flwr
from neural_structured_learning.keras.adversarial_regularization import AdversarialRegularization
import tensorflow as tf
from tensorflow import keras
from keras import layers
import neural_structured_learning as nsl
import tensorflow_datasets as tfds
from settings import *
from tensorflow.keras.utils import to_categorical
from keras.regularizers import l2

# todo: formulate robustness from metrics into formal math for paper ON paper

class HParams(object):
    '''
    adv_multiplier: The weight of adversarial loss in the training objective, relative to the labeled loss.
    adv_step_size: The magnitude of adversarial perturbation.
    adv_grad_norm: The norm to measure the magnitude of adversarial perturbation.

    Notes:
        - there are different regularization techniques, but keep technique constant
        - formalize relationship between adversarial input generation for a client and its server-side evaluation-under-attack.
        - adversarial regularization is very useful for defining an explicit structure e.g. structural signals rather than single samples.
        - nsl-ar structured signals provides more fine-grained information not available in feature inputs.
        - We can assume training with robust adversarial examples makes it robust against adversarial perturbations during inference (eval), but how does this process fair when there's a privacy-specific step of using client models to locally train on this data and use a federated optimization technique for server-side evaluation? How can we utilize unsupervised/semi-supervised learning and these "structured signals" to learn hidden representations in perturbed or otherwise corrupted data (image transformations) with applied gaussian noise (these configurations exist to simulate a real-world scenario). We want to then formalize this phenomenon and the results between each experimental configuration.
        - adv reg. --> how does this affect fed optimizer (regularized against adversarial attacks) and how would differences in fed optimizer affect adv. reg model? Seems like FedAdagrad is better on het. data, so if it was regularized anyway with adv. perturbation attacks, it should perform well against any uniform of non-uniform or non-bounded real-world or fixed norm perturbations.
        - wrap the adversarial regularization model to train under two other conditions relating to GaussianNoise and specified perturbation attacks during training specifically.
        - graph the feature representation given graph with respect to the graph of the rest of its computations, and the trusted aggregator eval
    '''

    def __init__(self, num_classes, adv_multiplier, adv_step_size, adv_grad_norm):
        self.input_shape = [28, 28, 1]
        self.num_classes = num_classes
        self.conv_filters = [32, 32, 64, 64, 128, 128, 256]
        self.kernel_size = (3, 3)
        self.pool_size = (2, 2)
        self.num_fc_units = [64]
        self.batch_size = 32
        self.epochs = 5
        self.adv_multiplier = adv_multiplier
        self.adv_step_size = adv_step_size
        self.adv_grad_norm = adv_grad_norm  # "l2" or "infinity" = l2_clip_norm if "l2"
        self.gaussian_state : bool = False
        # if (params.gaussian_state): model = tf.keras.models.Model.add(params.gaussian_layer, stack_index=1)
        self.gaussian_layer = keras.layers.GaussianNoise(stddev=0.2)
        self.clip_value_min = 0.0
        self.clip_value_max = 1.0

def build_base_model(params : HParams):
    input_layer = layers.Input(shape=(28, 28, 1), batch_size=None, name="image")
    conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
    batch_norm = layers.BatchNormalization()(conv1)
    dropout = layers.Dropout(0.3)(batch_norm)
    conv2 = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform',  padding='same')(dropout)
    maxpool1 = layers.MaxPool2D((2,2))(conv2)
    conv3 = layers.Conv2D(64, (3,3), activation='relu',  kernel_initializer='he_uniform', padding='same')(maxpool1)
    conv4 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv3)
    maxpool2 = layers.MaxPool2D((2,2))(conv4)
    conv5 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(maxpool2)
    conv6 = layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv5)
    maxpool3 = layers.MaxPool2D((2,2))(conv6)
    flatten = layers.Flatten()(maxpool3)
    # flatten is creating error because type : NoneType
    dense1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flatten)
    # possibly remove defined kernel/bias initializer, but functional API will check for this and removes error before processing model architecture and config
    output_layer = layers.Dense(params.num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')(dense1)
    model = keras.Model(inputs=input_layer, outputs=output_layer, name='client_model')
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model

def build_adv_model(params : HParams):
    input_layer = layers.Input(shape=(28, 28, 1), batch_size=None, name="image")
    conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
    batch_norm = layers.BatchNormalization()(conv1)
    dropout = layers.Dropout(0.3)(batch_norm)
    #  kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)
    conv2 = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform',padding='same')(dropout)
    maxpool1 = layers.MaxPool2D((2,2))(conv2)
    conv3 = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform',  padding='same')(maxpool1)
    conv4 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform',  padding='same')(conv3)
    maxpool2 = layers.MaxPool2D((2,2))(conv4)
    conv5 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform',  padding='same')(maxpool2)
    conv6 = layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_uniform',  padding='same')(conv5)
    maxpool3 = layers.MaxPool2D((2,2))(conv6)
    flatten = layers.Flatten()(maxpool3)
    dense1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flatten)
    # possibly remove defined kernel/bias initializer, but functional API will check for this and removes error before processing model architecture and config
    output_layer = layers.Dense(params.num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')(dense1)
    model = keras.Model(inputs=input_layer, outputs=output_layer, name='client_model')
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    adv_config = nsl.configs.make_adv_reg_config(multiplier=params.adv_multiplier, adv_step_size=params.adv_step_size, adv_grad_norm=params.adv_grad_norm)
    # AdvRegularization is a sub-class of tf.keras.Model, but it processes dicts instead for train and eval because of its decomposition approach for nsl
    adv_model = nsl.keras.AdversarialRegularization(model, label_keys=['label'], adv_config=adv_config, base_with_labels_in_features=True)
    adv_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return adv_model

def normalize(features):
  features['image'] = tf.cast(
      features['image'], dtype=tf.float32) / 255.0
  return features

def convert_to_tuples(features):
  return features['image'], features['label']

def convert_to_dictionaries(image, label):
  return {'image': image, 'label': label}

def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (
        x_train[idx * 5000 : (idx + 1) * 5000],
        y_train[idx * 5000 : (idx + 1) * 5000],
    ), (
        x_test[idx * 1000 : (idx + 1) * 1000],
        y_test[idx * 1000 : (idx + 1) * 1000],
    )

# prepare_experiment()

# create models
params = HParams(num_classes=10, adv_multiplier=0.2, adv_step_size=0.05, adv_grad_norm="infinity")
adv_model = build_adv_model(params=params)
base_model = build_base_model(params=params)
model = adv_model

IMAGE_INPUT_NAME = 'image'
LABEL_INPUT_NAME = 'label'
datasets = tfds.load('mnist')
train_dataset = datasets['train']
test_dataset = datasets['test']
train_dataset_for_base_model = train_dataset.map(normalize).shuffle(10000).batch(params.batch_size).map(convert_to_tuples)
test_dataset_for_base_model = test_dataset.map(normalize).batch(params.batch_size).map(convert_to_tuples)

# adv_model needs it to be processed in a dict, but flwr.client processes it as a tuple; this conflict is my error
train_set_for_adv_model = train_dataset_for_base_model.map(convert_to_dictionaries)
test_set_for_adv_model = test_dataset_for_base_model.map(convert_to_dictionaries)
# both datasets for the base and adv model should be of the same MapDataset type independent of adv_model feature dicts

# create_adv_client()
class AdvRegClient(flwr.client.NumPyClient):
    '''
    def __init__(self, args):
        self.train_dataset = args.current_train_partition
        self.test_dataset = args.current_train_partition
        # for using different models, try self.model = args.model
        self.params = args.params
    '''

    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(x=train_set_for_adv_model, epochs=5, steps_per_epoch=3, verbose=1)
        return model.get_weights(), len(train_set_for_adv_model), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        # unpack error
        loss, accuracy = model.evaluate(x=test_set_for_adv_model)
        return loss, len(test_set_for_adv_model), {"accuracy": accuracy}

class Client(flwr.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        # validation data param may be negligible
        history = model.fit(train_dataset_for_base_model, steps_per_epoch=3, epochs=5, verbose=1)
        # run the entire base model and check for its errors
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return model.get_weights(), results

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        # IterableDataset vs BatchDataset
        loss, accuracy = model.evaluate(test_dataset_for_base_model)
        return loss, {"accuracy": accuracy}

def main():
    # what entropy exists between the technique used for adv. reg. and the federated strategy given the data state per client
    # parser = argparse.ArgumentParser(description="entry point to client model configuration.")
    # # configurations
    # parser.add_argument("--num_partitions", type=int, choices=range(0, 100), required=True)
    # parser.add_argument("--adv_grad_norm", type=str, required=True)
    # parser.add_argument("--adv_multiplier", type=float, required=False, default=0.2)
    # parser.add_argument("--adv_step_size", type=float, choices=range(0, 1), required=True)
    # parser.add_argument("--batch_size", type=int, required=True)
    # parser.add_argument("--epochs", type=int, required=True)
    # parser.add_argument("--num_clients", type=int, required=True)
    # parser.add_argument("--num_rounds", type=int, required=True)
    # parser.add_argument("--federated_optimizer_strategy", type=str, required=True)
    # parser.add_argument("--adv_reg", type=bool, required=True)
    # parser.add_argument("--gaussian_layer", type=bool, required=True)
    # parser.add_argument("--fraction_fit", type=float, required=False, default=0.05)
    # parser.add_argument("--fraction_eval", type=float, required=False, default=0.5)
    # parser.add_argument("--min_fit_clients", type=int, required=False, default=10)
    # parser.add_argument("--min_eval_clients", type=int, required=False, default=2)
    # parser.add_argument("--min_available_clients", type=int, required=False, default=2)
    # parser.add_argument("--accept_client_failures_fault_tolerance", type=bool, required=False, default=False)
    # # weight regularization, SGD momentum --> other configs along with kernel/bias initializer
    # parser.add_argument("--weight_regularization", type=bool, required=False)
    # parser.add_argument("--sgd_momentum", type=float, required=False, default=0.9)
    # args = parser.parse_args()

    # start_client()
    if (type(model) == AdversarialRegularization):
        flwr.client.start_numpy_client("[::]:8080", client=AdvRegClient())

    elif (type(model) == tf.keras.models.Model):
        flwr.client.start_numpy_client("[::]:8080", client=Client())

if __name__ == '__main__':
    main()

