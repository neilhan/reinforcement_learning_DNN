import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import random_py_environment,tf_py_environment
from tf_agents.networks import encoding_network,network,utils
from tf_agents.specs import array_spec
from tf_agents.utils import common as common_utils, nest_utils

tf.compat.v1.enable_v2_behavior()