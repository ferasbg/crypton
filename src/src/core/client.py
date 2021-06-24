import argparse
import flwr as fl
import tensorflow as tf
from tensorflow import keras
from keras import layers
import neural_structured_learning as nsl

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

    def __init__(self, num_classes, adv_multiplier, adv_step_size, adv_grad_norm, state):
        self.input_shape = [32, 32, 3]
        self.num_classes = num_classes
        self.conv_filters = [32, 64, 64, 128, 128, 256]
        self.kernel_size = (3, 3)
        self.pool_size = (2, 2)
        self.num_fc_units = [64]
        self.batch_size = 32
        self.epochs = 5
        self.adv_multiplier = adv_multiplier
        self.adv_step_size = adv_step_size
        self.adv_grad_norm = adv_grad_norm  # "l2" or "infinity" = l2_clip_norm if "l2"
        self.client_regularization_state : bool = state
        self.gaussian_state : bool = False
        # if gaussian_state: append after input layer at index 1
        self.gaussian_layer = keras.layers.GaussianNoise(stddev=0.2)

def build_base_model(parameters : HParams):
    input_layer = layers.Input(shape=(32,32,3), batch_size=None, name="image") 
    conv1 = layers.Conv2D(32, parameters.kernel_size, activation='relu', padding='same')(input_layer)
    batch_norm = layers.BatchNormalization()(conv1)
    dropout = layers.Dropout(0.3)(batch_norm)
    conv2 = layers.Conv2D(64, parameters.kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same')(dropout)
    maxpool1 = layers.MaxPool2D(parameters.pool_size)(conv2)
    conv3 = layers.Conv2D(64, parameters.kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same')(maxpool1)
    conv4 = layers.Conv2D(128, parameters.kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same')(conv3)
    maxpool2 = layers.MaxPool2D(parameters.pool_size)(conv4)
    conv5 = layers.Conv2D(128, parameters.kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same')(maxpool2)
    conv6 = layers.Conv2D(256, parameters.kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same')(conv5)
    maxpool3 = layers.MaxPool2D(parameters.pool_size)(conv6)
    flatten = layers.Flatten()(maxpool3)
    # flatten is creating error because type : NoneType
    dense1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flatten)
    # possibly remove defined kernel/bias initializer, but functional API will check for this and removes error before processing model architecture and config
    output_layer = layers.Dense(parameters.num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')(dense1)
    model = keras.Model(inputs=input_layer, outputs=output_layer, name='client_model')
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model

def build_adv_model(parameters : HParams):
    input_layer = layers.Input(shape=(32,32,3), batch_size=None, name="image") 
    conv1 = layers.Conv2D(32, parameters.kernel_size, activation='relu', padding='same')(input_layer)
    batch_norm = layers.BatchNormalization()(conv1)
    dropout = layers.Dropout(0.3)(batch_norm)
    conv2 = layers.Conv2D(64, parameters.kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same')(dropout)
    maxpool1 = layers.MaxPool2D(parameters.pool_size)(conv2)
    conv3 = layers.Conv2D(64, parameters.kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same')(maxpool1)
    conv4 = layers.Conv2D(128, parameters.kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same')(conv3)
    maxpool2 = layers.MaxPool2D(parameters.pool_size)(conv4)
    conv5 = layers.Conv2D(128, parameters.kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same')(maxpool2)
    conv6 = layers.Conv2D(256, parameters.kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same')(conv5)
    maxpool3 = layers.MaxPool2D(parameters.pool_size)(conv6)
    flatten = layers.Flatten()(maxpool3)
    # flatten is creating error because type : NoneType
    dense1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flatten)
    # possibly remove defined kernel/bias initializer, but functional API will check for this and removes error before processing model architecture and config
    output_layer = layers.Dense(parameters.num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')(dense1)
    model = keras.Model(inputs=input_layer, outputs=output_layer, name='client_model')
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    adv_config = nsl.configs.make_adv_reg_config(multiplier=parameters.adv_multiplier, adv_step_size=parameters.adv_step_size, adv_grad_norm=parameters.adv_grad_norm)
    # AdvRegularization is a sub-class of tf.keras.Model, so processing the data to input layer in train/test may differ (probably not though)
    model = nsl.keras.AdversarialRegularization(model, label_keys=['label'], adv_config=adv_config, base_with_labels_in_features=True)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

parameters = HParams(num_classes=10, adv_multiplier=0.2, adv_step_size=0.05, adv_grad_norm="infinity", state=True)

# ad-hoc config
if (parameters.client_regularization_state == True):
    model = build_adv_model(parameters=parameters)

if (parameters.client_regularization_state == False):
    model = build_base_model(parameters=parameters)

# partition data here, perturb batches here, apply corruptions here; everything done before it's processed to Client
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# train_data = tf.data.Dataset.from_tensor_slices({'image': x_train, 'label': y_train}).batch(batch_size=32)
# val_data = tf.data.Dataset.from_tensor_slices({'image': x_test, 'label': y_test}).batch(batch_size=32)
# val_steps = x_test.shape[0] / 32 

# define functions to convert between dicts and tuples

IMAGE_INPUT_NAME = 'image'
LABEL_INPUT_NAME = 'label'

def normalize(features):
  features[IMAGE_INPUT_NAME] = tf.cast(
      features[IMAGE_INPUT_NAME], dtype=tf.float32) / 255.0
  return features

def convert_to_tuples(features):
  return features[IMAGE_INPUT_NAME], features[LABEL_INPUT_NAME]

def convert_to_dictionaries(image, label):
  return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}

# train_dataset = train_data.map(normalize).shuffle(10000).batch(parameters.batch_size).map(convert_to_tuples)
# test_dataset = val_data.map(normalize).batch(parameters.batch_size).map(convert_to_tuples)
# use MNIST?

class Client(fl.client.NumPyClient):
    def get_parameters(self):  # type: ignore
        return model.get_weights()

    def fit(self, parameters, config):  # type: ignore
        model.set_weights(parameters)
        # use dict to test Client
        model.fit(x={'image': x_train, 'label': y_train}, epochs=parameters.epochs, verbose=1)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        model.set_weights(parameters)
        # replicate error
        loss, accuracy = model.evaluate(x={'image': x_test, 'label': y_test})
        return loss, len(x_test), {"accuracy": accuracy}

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