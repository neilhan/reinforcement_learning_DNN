import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import random_py_environment, tf_py_environment
from tf_agents.networks import encoding_network, network, utils
from tf_agents.specs import array_spec
from tf_agents.utils import common as common_utils, nest_utils

tf.compat.v1.enable_v2_behavior()


class CustomNN8x8(network.Network):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 name='othelloActorNetwork'):
        super().__init__(input_tensor_spec=observation_spec, state_spec=(), name=name)
        self._action_spec = action_spec
        flat_action_spec = tf.nest.flatten(action_spec)
        self._single_action_spec = flat_action_spec[0]

        self._flat_x = tf.keras.layers.Flatten()
        self._cnn_1 = self._create_conv2d_layer(32, 2, 1)
        self._cnn_2 = self._create_conv2d_layer(64, 2, 1)
        self._flat_cnn = tf.keras.layers.Flatten() 
        # will shortcut here
        self._dense_1 = self._create_dense_layer(1024)
        self._dense_2 = self._create_dense_layer(512)
        self._policy_dense_1 = self._create_dense_layer(128)

        initializer = tf.keras.initializers.RandomUniform(
            minval=-0.003, maxval=0.003)
        self._action_projection_layer = tf.keras.layers.Dense(action_spec.maximum + 1,
                                                            #   activation=tf.keras.activations.tanh,
                                                              kernel_initializer=initializer,
                                                              name='action')
        # this is only doing one layer for test the Custom model.

    def call(self, observations, step_type=(), network_state=()):
        outer_rank = nest_utils.get_outer_rank(
            observations, self.input_tensor_spec)
        # batch_squash, in case observations have a time sequence compoment.
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(
            batch_squash.flatten, observations)

        flat_x = self._flat_x(observations)
        cnn_1 = self._cnn_1(observations)
        cnn_2 = self._cnn_2(cnn_1)
        flat_cnn = self._flat_cnn(cnn_2)
        concat_cnn_x = tf.keras.layers.concatenate([flat_x, flat_cnn])
        dense_1 = self._dense_1(concat_cnn_x)
        dense_2 = self._dense_2(dense_1)
        policy_dense_1 = self._policy_dense_1(dense_2)
        actions = self._action_projection_layer(policy_dense_1)

        return tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state

    @staticmethod
    def _create_conv2d_layer(num_filters, kernel_size, strides):
        return tf.keras.layers.Conv2D(filters=num_filters,
                                      strides=(strides, strides),
                                      kernel_size=kernel_size,
                                      activation=tf.nn.relu,
                                      kernel_initializer='random_normal',
                                      )

    @staticmethod
    def _create_dense_layer(num_nodes, act_fn=tf.nn.relu):
        return tf.keras.layers.Dense(units=num_nodes, activation=act_fn)


class CustomNN(network.Network):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 name='othelloActorNetwork'):
        super().__init__(input_tensor_spec=observation_spec, state_spec=(), name=name)
        self._action_spec = action_spec
        flat_action_spec = tf.nest.flatten(action_spec)
        self._single_action_spec = flat_action_spec[0]

        self._flat_x = tf.keras.layers.Flatten()
        self._cnn_1 = self._create_conv2d_layer(32, 2, 1)
        self._flat_cnn = tf.keras.layers.Flatten() 
        # will shortcut here
        self._dense_1 = self._create_dense_layer(128)
        self._policy_dense_1 = self._create_dense_layer(64)

        initializer = tf.keras.initializers.RandomUniform(
            minval=-0.003, maxval=0.003)
        self._action_projection_layer = tf.keras.layers.Dense(action_spec.maximum + 1,
                                                            #   activation=tf.keras.activations.tanh,
                                                              kernel_initializer=initializer,
                                                              name='action')
        # this is only doing one layer for test the Custom model.

    def call(self, observations, step_type=(), network_state=()):
        outer_rank = nest_utils.get_outer_rank(
            observations, self.input_tensor_spec)
        # batch_squash, in case observations have a time sequence compoment.
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(
            batch_squash.flatten, observations)

        flat_x = self._flat_x(observations)
        cnn = self._cnn_1(observations)
        flat_cnn = self._flat_cnn(cnn)
        concat_cnn_x = tf.keras.layers.concatenate([flat_x, flat_cnn])
        dense_1 = self._dense_1(concat_cnn_x)
        policy_dense_1 = self._policy_dense_1(dense_1)
        actions = self._action_projection_layer(policy_dense_1)

        return tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state

    @staticmethod
    def _create_conv2d_layer(num_filters, kernel_size, strides):
        return tf.keras.layers.Conv2D(filters=num_filters,
                                      strides=(strides, strides),
                                      kernel_size=kernel_size,
                                      activation=tf.nn.relu,
                                      kernel_initializer='random_normal',
                                      )

    @staticmethod
    def _create_dense_layer(num_nodes, act_fn=tf.nn.relu):
        return tf.keras.layers.Dense(units=num_nodes, activation=act_fn)


if __name__ == '__main__':
    action_spec = array_spec.BoundedArraySpec(
        (3,), np.float32, minimum=0, maximum=10)
    observation_spec = {
        'image': array_spec.BoundedArraySpec((16, 16, 3), np.float32, minimum=0,
                                             maximum=255),
        'vector': array_spec.BoundedArraySpec((5,), np.float32, minimum=-100,
                                              maximum=100)}

    random_env = random_py_environment.RandomPyEnvironment(
        observation_spec, action_spec=action_spec)

    # Convert the environment to a TFEnv to generate tensors.
    tf_env = tf_py_environment.TFPyEnvironment(random_env)
