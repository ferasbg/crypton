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
    '''

    def __init__(self, num_classes, adv_multiplier, adv_step_size, adv_grad_norm):
        self.input_shape = [28, 28, 1]
        self.num_classes = num_classes
        self.conv_filters = [32, 64, 64, 128, 128, 256]
        self.kernel_size = (3, 3)
        self.pool_size = (2, 2)
        self.num_fc_units = [64]
        self.batch_size = 32
        self.epochs = 5
        self.adv_multiplier = adv_multiplier
        self.adv_step_size = adv_step_size
        self.adv_grad_norm = adv_grad_norm  # "l2" or "infinity"

def normalize(features):
  features['image'] = tf.cast(
      features['image'], dtype=tf.float32) / 255.0
  return features

def convert_to_tuples(features):
  return features['image'], features['label']

def convert_to_dictionaries(image, label, IMAGE_INPUT_NAME='image', LABEL_INPUT_NAME='label'):
  return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}


def build_model(params: HParams, num_classes: int):
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

params = HParams(num_classes=10, adv_multiplier=0.2,
                     adv_step_size=0.05, adv_grad_norm="infinity")

adv_model = build_model(params=params, num_classes=10)

# test with MapDataset for both base and adv model --> then to partitions
datasets = tfds.load('mnist')
train_dataset = datasets['train']
test_dataset = datasets['test']

train_dataset_for_base_model = train_dataset.map(normalize).shuffle(10000).batch(params.batch_size).map(convert_to_tuples)
test_dataset_for_base_model = test_dataset.map(normalize).batch(params.batch_size).map(convert_to_tuples)

# type: MapDataset
train_set_for_adv_model = train_dataset_for_base_model.map(convert_to_dictionaries)
test_set_for_adv_model = test_dataset_for_base_model.map(convert_to_dictionaries)

# manually perturb the data per batch, since adv_model requires a different format unless you want to convert the format back into base_dataset from adv_model_dataset 
print(type(train_dataset_for_base_model))
print(type(test_dataset_for_base_model))

for batch in train_set_for_adv_model:
    adv_model.perturb_on_batch(batch)

for batch in test_set_for_adv_model:
    adv_model.perturb_on_batch(batch)

base_model_train_dataset = train_set_for_adv_model.batch(params.batch_size).map(convert_to_tuples)
base_model_test_dataset = test_set_for_adv_model.batch(params.batch_size).map(convert_to_tuples)

print(type(base_model_train_dataset))
print(type(base_model_test_dataset))