import logging
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


class A2CModel:
    """Combined actor-critic network."""

    def __init__(self,
                 vision_shape=(8, 8, 1),
                 num_actions: int = 8*8+1):
        """Initialize."""
        super().__init__()

        self.num_actions = num_actions
        self.vision_shape = vision_shape

        with tf.name_scope('model'):
            X = tf.keras.Input(shape=vision_shape, dtype=tf.dtypes.float32)
            X_normal = X / 4.0 + 1
            X_normal_flat = tf.keras.layers.Flatten()(X_normal)
            # fork: to policy and value_fn
            cnn_1 = self._create_conv2d_layer(32, 2, 2)(X_normal)
            cnn_2 = self._create_conv2d_layer(32, 2, 1)(cnn_1)
            flat_layer_3 = tf.keras.layers.Flatten()(cnn_2)
            concat_X_cnn = tf.keras.layers.concatenate(
                [X_normal_flat, flat_layer_3])
            policy_dense_4 = self._create_dense_layer(64)(concat_X_cnn)
            policy_logits = self._create_dense_layer(num_actions,
                                                     act_fn=None)(policy_dense_4)
            value_dense_4 = self._create_dense_layer(32)(flat_layer_3)
            value_fn = self._create_dense_layer(1, act_fn=None)(value_dense_4)

        self.policy_logits = policy_logits
        self.value_fn = value_fn

        self._model = \
            tf.keras.Model(inputs=X,
                           outputs=[self.policy_logits,
                                    self.value_fn],
                           name='A2CModel') 

    def call(self, X: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        policy_logits, value = self._model(X)
        return policy_logits, value

    def save_model(self, model_path):
        self._model.save(model_path)
    
    def load_model(self, model_path):
        self._model=tf.keras.models.load_model(model_path)

    @staticmethod
    def _create_conv2d_layer(num_filters, kernel_size, strides):
        return tf.keras.layers.Conv2D(filters=num_filters,
                                      strides=(strides, strides),
                                      kernel_size=kernel_size,
                                      activation=tf.nn.relu,
                                      kernel_initializer='random_normal')

    @staticmethod
    def _create_dense_layer(num_nodes, act_fn=tf.nn.relu):
        return tf.keras.layers.Dense(units=num_nodes, activation=act_fn)
    


if __name__ == '__main__':
    # create new instance
    model = A2CModel()

    action, values = \
        model.get_action_value(np.zeros((5, 8, 8, 1), dtype=np.float32), [])

    print('action:', action)
    print('action:', tf.make_ndarray(tf.make_tensor_proto(action)))
    print('values:', values)
    print('values:', tf.make_ndarray(tf.make_tensor_proto(values)))

    model.summary()
