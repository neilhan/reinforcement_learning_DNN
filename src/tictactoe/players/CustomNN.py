import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import random_py_environment, tf_py_environment
from tf_agents.networks import encoding_network, network, utils
from tf_agents.specs import array_spec
from tf_agents.utils import common as common_utils, nest_utils

tf.compat.v1.enable_v2_behavior()


class CustomNN(network.Network):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 preprocessing_layers=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=(75, 40),
                 dropout_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 enable_last_layer_zero_initializer=False,
                 name='ActorNetwork'):
        super().__init__(input_tensor_spec=observation_spec, state_spec=(), name=name)
        self._action_spec = action_spec
        flat_action_spec = tf.nest.flatten(action_spec)
        self._single_action_spec = flat_action_spec[0]

        kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1.0/3.0, mode='fan_in', distribution='uniform')
        self._encoder = encoding_network.EncodingNetwork(observation_spec,
                                                         preprocessing_layers=preprocessing_layers,
                                                         preprocessing_combiner=preprocessing_combiner,
                                                         conv_layer_params=conv_layer_params,
                                                         fc_layer_params=fc_layer_params,
                                                         dropout_layer_params=dropout_layer_params,
                                                         activation_fn=activation_fn,
                                                         kernel_initializer=kernel_initializer,
                                                         batch_squash=False)

        initializer = tf.keras.initializers.RandomUniform(
            minval=-0.003, maxval=0.003)
        self._action_projection_layer = tf.keras.layers.Dense(flat_action_spec[0].shape.num_elements(),
                                                              activation=tf.keras.activations.tanh,
                                                              kernel_initializer=initializer,
                                                              name='action')

    def call(self, observations, step_type=(), network_state=()):
        outer_rank = nest_utils.get_outer_rank(
            observations, self.input_tensor_spec)
        # batch_squash, in case observations have a time sequence compoment.
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(
            batch_squash.flatten, observations)

        state, network_state = self._encoder(observations,
                                             step_type=step_type,
                                             network_state=network_state)
        actions = self._action_projection_layer(state)
        actions = common_utils.scale_to_spec(actions, self._single_action_spec)
        actions = batch_squash.unflatten(actions)
        return tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state


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
