import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops


def set_global_seeds(seed: int):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def rot90_4d_batch(images):
    """
    When the tensor is in shape: [batch_idx, h, w, channels],
    This function rotate images counter clock 90 degrees
    """
    return array_ops.transpose(array_ops.reverse_v2(images, [2]), [0, 2, 1, 3])


def rot90_5d_batch(images):
    """
    When the tensor is in shape: [batch_idx, #parallel_envs, h, w, channels],
    This function rotate images counter clock 90 degrees
    """
    return array_ops.transpose(array_ops.reverse_v2(images, [3]), [0, 1, 3, 2, 4])


