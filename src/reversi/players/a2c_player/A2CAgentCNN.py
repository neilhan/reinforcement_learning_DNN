import numpy as np
import tensorflow as tf


class VisionShape:
    def __init__(self, height: int, width: int,  depth: int):
        self.height = height
        self.width = width
        self.depth = depth  # number of coler layers


class A2CAgentCNN:
    def __init__(self,
                 batch_size: int,
                 vision_shape: VisionShape,
                 action_shape: int, # we have just one dimension. 
                 time_window_size: int = 1):

        features_shape = (vision_shape.height,
                          vision_shape.width,
                          vision_shape.depth * time_window_size)
        X = tf.keras.Input(shape=features_shape, batch_size=batch_size)
        # hardcode the normalization scale for now. 1/2, to 0.5 or -0.5
        X_normal = tf.cast(X, tf.float32) / 2.0

        # create the network
        with tf.name_scope('model') as scope:
            c1 = self._create_conv2d_layer(32, 4, 1)(X_normal)
            c2 = self._create_conv2d_layer(64, 4, 1)(c1)
            flat_layer_3 = tf.keras.layers.Flatten()(c2)
            dense_4 = self._create_dense_layer(512)(flat_layer_3)
            # fork: to policy and value_fn
            policy_fn = self._create_dense_layer(
                action_shape, act_fn=None)(dense_4)
            value_fn = self._create_dense_layer(1, act_fn=None)(dense_4)

        # only need the first input/output, so get v0, and a0
        self.v0 = value_fn[:, 0]
        self.a0 = self._sample(policy_fn)

        self._model = tf.keras.Model(inputs=X,
                                     outputs=[policy_fn, value_fn],
                                     name='A2CAgentCNN')

    @tf.function
    def step(self, observation):
        # observation: in shape of: (width, height, depth * time_window_size)
        # return (actions, value) only one output
        # Should ???? observation should be padded to batch size, depth, time_window_size
        output_actions, output_values = self._model.call(observation)
        return (self._sample(output_actions), output_values[:, 0])

    @tf.function
    def value(self, observation):
        # observation: in shape of: (width, height, depth * time_window_size)
        _, output_values = self._model.call(observation)
        return output_values[:, 0]

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
    @staticmethod
    def _sample(logits):
        noise = tf.random.uniform(tf.shape(logits))
        return tf.argmax(logits - tf.math.log(-tf.math.log(noise)), 1)


if __name__ == '__main__':
    # create new instance
    agent_cnn = A2CAgentCNN(5, VisionShape(8, 8, 1), 8*8, time_window_size=1)
    agent_cnn._model.summary()
