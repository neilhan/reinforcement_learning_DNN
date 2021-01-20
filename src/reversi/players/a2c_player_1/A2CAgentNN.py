from __future__ import annotations

import random
import numpy as np
import tensorflow as tf
import logging

from reversi.players.a2c_player_1.GameWrapper import GameWrapper, PASS_TURN
from reversi.game import GameBoard


class A2CAgentNN:
    def __init__(self, input_size: int = 8*8, action_size: int = (8*8 + 1)):
        # the action 0, to 63 are the moves to take. The action 64 is pass this turn.

        # create the network
        with tf.name_scope('model'):
            X = tf.keras.Input(shape=(input_size,), dtype=tf.dtypes.float32)
            # fork: to policy and value_fn
            policy_dense_1 = self._create_dense_layer(512)(X)
            policy_dense_2 = self._create_dense_layer(512)(policy_dense_1)
            policy_dense_3 = self._create_dense_layer(512)(policy_dense_2)
            policy_logits = self._create_dense_layer(action_size,
                                                     act_fn=None)(policy_dense_3)
            value_dense_1 = self._create_dense_layer(512)(X)
            value_dense_2 = self._create_dense_layer(512)(value_dense_1)
            value_dense_3 = self._create_dense_layer(512)(value_dense_2)
            value_fn = self._create_dense_layer(1, act_fn=None)(value_dense_3)

        self.policy_logits = policy_logits
        self.value_fn = value_fn

        self._model = tf.keras.Model(inputs=X,
                                     outputs=[self.policy_logits,
                                              self.value_fn],
                                     name='A2CAgent')

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

    # logits is the network policy output
    # sample the next action
    # (logits - log(-log(noise)) to introduce random action
    @tf.function
    def _sample_model(self, obs):
        output_action_logits, output_values = self._model.call(obs)
        noise = tf.random.uniform(tf.shape(output_action_logits))
        sampled_action = tf.argmax(
            output_action_logits - tf.math.log(-tf.math.log(noise)), 1)
        return sampled_action, output_values

    def get_action_value(self, observation, all_valid_moves, force_valid=False):
        sampled_action, output_values = self._sample_model(observation)
        # # This following logic is when forcing the action to a valid move
        if force_valid:
            if len(all_valid_moves) > 0 and not sampled_action in all_valid_moves:
                logging.debug('A2CAgentNN.get_action_value() pick random action')
                sampled_action = random.choice(all_valid_moves)
            elif len(all_valid_moves) == 0 and sampled_action != PASS_TURN:
                logging.debug(
                    'A2CAgentNN.get_action_value() forced correction pass')
                sampled_action = PASS_TURN

        return (sampled_action, output_values[:, 0])
        # return (actions, value) only one output



if __name__ == '__main__':
    # create new instance
    agent_nn = A2CAgentNN()
    action, values = \
        agent_nn.get_action_value(np.zeros((15, 8*8), dtype=np.float32), [])

    # tf.config.run_functions_eagerly(True)
    # print('run function eagerly:', tf.executing_eagerly())

    print('action:', action)
    print('action:', tf.make_ndarray(tf.make_tensor_proto(action)))
    print('values:', values)
    print('values:', tf.make_ndarray(tf.make_tensor_proto(values)))

    agent_nn._model.summary()
