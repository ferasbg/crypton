import argparse
import flwr as fl
import tensorflow as tf
from tensorflow import keras
from keras import layers
import neural_structured_learning as nsl
import tensorflow_datasets as tfds

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
        self.gaussian_state : bool = False
        # if gaussian_state: append after input layer at index 1
        self.gaussian_layer = keras.layers.GaussianNoise(stddev=0.2)

def build_base_model(parameters : HParams):
    input_layer = layers.Input(shape=(28,28,1), batch_size=None, name="image")
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
    dense1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flatten)
    output_layer = layers.Dense(parameters.num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')(dense1)
    model = keras.Model(inputs=input_layer, outputs=output_layer, name='client_model')
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model

def build_adv_model(parameters : HParams):
    input_layer = layers.Input(shape=(28,28,1), batch_size=None, name="image")
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
    dense1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flatten)
    output_layer = layers.Dense(parameters.num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')(dense1)
    model = keras.Model(inputs=input_layer, outputs=output_layer, name='client_model')
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    adv_config = nsl.configs.make_adv_reg_config(multiplier=parameters.adv_multiplier, adv_step_size=parameters.adv_step_size, adv_grad_norm=parameters.adv_grad_norm)
    model = nsl.keras.AdversarialRegularization(model, label_keys=['label'], adv_config=adv_config, base_with_labels_in_features=True)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

parameters = HParams(num_classes=10, adv_multiplier=0.2, adv_step_size=0.05, adv_grad_norm="infinity")

# ad-hoc config to create client model
adv_model = build_adv_model(parameters=parameters)
base_model = build_base_model(parameters=parameters)

# partition data here, perturb batches here, apply corruptions here; everything done before it's processed to Client
IMAGE_INPUT_NAME = 'image'
LABEL_INPUT_NAME = 'label'

datasets = tfds.load('mnist')
train_dataset = datasets['train']
test_dataset = datasets['test']

# define functions to convert between dicts and tuples
def normalize(features):
  features[IMAGE_INPUT_NAME] = tf.cast(
      features[IMAGE_INPUT_NAME], dtype=tf.float32) / 255.0
  return features

def convert_to_tuples(features):
  return features[IMAGE_INPUT_NAME], features[LABEL_INPUT_NAME]

def convert_to_dictionaries(image, label):
  return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}

train_dataset = train_dataset.map(normalize).shuffle(10000).batch(parameters.batch_size).map(convert_to_tuples)
test_dataset = test_dataset.map(normalize).batch(parameters.batch_size).map(convert_to_tuples)

# base_model.fit(train_dataset, epochs=parameters.epochs)
# results = base_model.evaluate(test_dataset)

adv_model = build_adv_model(parameters=parameters)
# adv_model needs it to be processed in a dict, but flwr.client processes it as a tuple; this conflict is my error
train_set_for_adv_model = train_dataset.map(convert_to_dictionaries)
test_set_for_adv_model = test_dataset.map(convert_to_dictionaries)


# contradiction: adv train set works fine, but .evaluate can't unpack the data that was already processed
# the types are normal; base model works fine with tuples, but unpacking dicts seems to create the error
# best workaround is to see what types are accepted for clients rather than nsl itself
# the requirements for the data are different with flwr; that being said: this is more of a data formatting error and types error as a result of using nsl and flwr


#adv_history = adv_model.fit(train_set_for_adv_model, steps_per_epoch=3, epochs=parameters.epochs)
#print(adv_history)
adv_results = adv_model.evaluate(test_set_for_adv_model)
# default: IID data given artificial setup of the clients and the data
