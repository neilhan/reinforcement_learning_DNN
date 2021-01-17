import numpy as np
import tensorflow as tf

def set_global_seeds(seed:int):
    tf.random.set_seed(seed)
    np.random.seed(seed)