import keras
import os

import numpy as np
import random
import time
from tqdm import tqdm
from collections import deque

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam



