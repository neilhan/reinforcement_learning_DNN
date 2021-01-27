import logging
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


class A2CModel(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self,
                 num_actions: int = 8*8+1):
        """Initialize."""
        super().__init__()

        self.num_actions = num_actions

        self.X_normal_flat = tf.keras.layers.Flatten()
        # fork: to policy and value_fn
        self.cnn_1 = self._create_conv2d_layer(30, 2, 2)
        self.cnn_2 = self._create_conv2d_layer(30, 2, 1)
        self.flat_layer_3 = tf.keras.layers.Flatten()
        self.concat_X_cnn = tf.keras.layers.Concatenate()
        # fork happens from here. After concat, -> policy, or -> value
        self.policy_dense_4 = self._create_dense_layer(64)
        self.policy_logits = self._create_dense_layer(num_actions, act_fn=None)

        self.value_dense_4 = self._create_dense_layer(32)
        self.value_fn = self._create_dense_layer(1, act_fn=None)


    def call(self, X: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        X_normal = X / 4.0 + 1
        X_normal_flat = self.X_normal_flat(X_normal)
        cnn_1 = self.cnn_1(X_normal)
        cnn_2 = self.cnn_2(cnn_1)
        flat_3 = self.flat_layer_3(cnn_2)
        concat_X_cnn = self.concat_X_cnn([X_normal_flat, flat_3])
        policy_dense_4 = self.policy_dense_4(concat_X_cnn)
        policy_logits = self.policy_logits(policy_dense_4)

        value_dense_4 = self.value_dense_4(concat_X_cnn)
        value = self.value_fn(value_dense_4)

        return policy_logits, value 

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
