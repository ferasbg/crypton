# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Partitioned version of CIFAR-10 dataset."""

from typing import List, Tuple, cast

import numpy as np
import tensorflow as tf


def shuffle(x: np.ndarray, y: np.ndarray):
    """Shuffle x and y."""
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


def partition(x: np.ndarray, y: np.ndarray, num_partitions: int):
    """Split x and y into a number of partitions."""
    return list(zip(np.split(x, num_partitions), np.split(y, num_partitions)))


def create_partitions(
    source_dataset,
    num_partitions: int,
):
    """Create partitioned version of a source dataset."""
    # they expect XY which is a Tuple[np.ndarray, np.ndarray] so it's either train_data or val_data
    x, y = source_dataset
    # x,y are ndarrays inside the Tuple; cannot have a tuple of tuples
    x, y = shuffle(x, y)
    xy_partitions = partition(x, y, num_partitions)

    return xy_partitions


def load(num_partitions: int,):
    
    """Create partitioned version of CIFAR-10."""
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # adv_model perturbs the batches when the batch is processed; what if we perturb before?
    xy_train = (x_train, y_train)
    xy_test = (x_test, y_test)

    # unpack into feature tuple, and not a list of tuples
    xy_train_partitions = create_partitions(xy_train, num_partitions)
    xy_test_partitions = create_partitions(xy_test, num_partitions)

    # list(zip) is the format of partition data; perhaps it's not able to unpack a dict when it's partitioned with list(zip(dataset_partition based on clients))
    return list(zip(xy_train_partitions, xy_test_partitions))
