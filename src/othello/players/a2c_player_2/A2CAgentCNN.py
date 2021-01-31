from __future__ import annotations

import random
import numpy as np
import tensorflow as tf
import logging

from othello.players.a2c_player_2.GameWrapper import GameWrapper
from othello.game import GameBoard


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        # Sample a random categorical action from the given logits
        return tf.squeeze(tf.random.categorical(tf.math.log(logits), 1), axis=-1)
        # return tf.squeeze(tf.random.categorical((logits), 1), axis=-1)


class A2CAgentCNN:
    def __init__(self, vision_shape=(8, 8, 1), num_actions: int = (8*8 + 1)):
        # the action 0, to 63 are the moves to take. The action 64 is pass this turn.
        self.PASS_TURN_ACTION = num_actions  # the last one is pass to other player

        # create the network
        with tf.name_scope('model'):
            X = tf.keras.Input(shape=vision_shape, dtype=tf.dtypes.float32)
            X_normal = X / 4.0 + 1
            X_normal_flat = tf.keras.layers.Flatten()(X_normal)
            # fork: to policy and value_fn
            cnn_1 = self._create_conv2d_layer(50, 2, 2)(X_normal)
            cnn_2 = self._create_conv2d_layer(50, 2, 1)(cnn_1)
            flat_layer_3 = tf.keras.layers.Flatten()(cnn_2)
            concat_X_cnn = tf.keras.layers.concatenate([X_normal_flat, flat_layer_3])
            policy_dense_4 = self._create_dense_layer(128)(concat_X_cnn)
            policy_logits = self._create_dense_layer(num_actions,
                                                     act_fn=None)(policy_dense_4)
            value_dense_4 = self._create_dense_layer(32)(flat_layer_3)
            value_fn = self._create_dense_layer(1, act_fn=None)(value_dense_4)

        self.policy_logits = policy_logits
        self.value_fn = value_fn
        self.dist = ProbabilityDistribution()
        self.vision_shape = vision_shape

        self._model = \
            tf.keras.Model(inputs=X,
                           outputs=[self.policy_logits,
                                    self.value_fn],
                           name='A2CAgentCNN')

    # logits is the network policy output
    # sample the next action
    # (logits - log(-log(noise)) to introduce random action
    @tf.function
    def _sample_model(self, obs):
        output_action_logits, output_values = self._model.call(obs)
        return output_action_logits, output_values

    def get_action_value(self, observation, all_valid_moves, force_valid=False):
        action_logits, output_values = self._sample_model(observation)
        sampled_action = self.dist.predict(action_logits)
        # # This following logic is when forcing the action to a valid move
        if force_valid:
            if len(all_valid_moves) > 0 and not sampled_action in all_valid_moves:
                logging.debug(
                    'A2CAgentCNN.get_action_value() pick random action')
                sampled_action = random.choice(all_valid_moves)
            elif len(all_valid_moves) == 0 and sampled_action != self.PASS_TURN_ACTION:
                logging.debug(
                    'A2CAgentCNN.get_action_value() forced correction pass')
                sampled_action = self.PASS_TURN_ACTION

        return (sampled_action, output_values[:, 0])
        # return (actions, value) only one output

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
    
    def save_model(self, model_path):
        self._model.save(model_path)
    
    def load_model(self, model_path):
        self._model=tf.keras.models.load_model(model_path)



if __name__ == '__main__':
    # create new instance
    agent_nn = A2CAgentCNN()
    action, values = \
        agent_nn.get_action_value(
            np.zeros((5, 8, 8, 1), dtype=np.float32), [])

    # tf.config.run_functions_eagerly(True)
    # print('run function eagerly:', tf.executing_eagerly())

    print('action:', action)
    print('action:', tf.make_ndarray(tf.make_tensor_proto(action)))
    print('values:', values)
    print('values:', tf.make_ndarray(tf.make_tensor_proto(values)))

    agent_nn._model.summary()
