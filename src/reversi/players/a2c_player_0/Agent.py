import os
import numpy as np
import tensorflow as tf

from reversi.players.a2c_player.A2CAgentCNN import VisionShape, A2CAgentCNN

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class Agent:
    # is the Agent.
    # Agent can be trained. saves its NN, loads NN etc.
    def __init__(self, NN_class, vision_shape: VisionShape,
                 num_workers, num_steps, time_window_size: int = 1,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(10e3)):
        # TODO threads
        # tf.config.threading.set_intra_op_parallelism_threads(0)  # let system decide
        # tf.config.threading.set_inter_op_parallelism_threads(0)  # let system decide
        # gpu_devices = tf.config.list_physical_devices('GPU')
        # if len(gpu_devices) > 0:
        # tf.config.experimental.set_memory_growth(gpu_devices[0], True)

        self.nn = NN_class(batch_size=5,
                           vision_shape=vision_shape,
                           action_shape=(8*8))
        batch_size = num_workers * num_steps

        A = tf.keras.Input(batch_size=batch_size, shape=[1], dtype=tf.dtypes.int32)
        ADV = tf.keras.Input([batch_size], dtype=tf.dtypes.float32)
        R = tf.keras.Input([batch_size], dtype=tf.dtypes.float32)
        LR = tf.keras.Input([], dtype=tf.dtypes.float32)

        # build the loss_fn
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=A, logits=self.nn.policy_fn)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(tf.math.squared_differenc(
            tf.squeeze(self.nn.value_fn), R) / 2.0)
        entropy = tf.reduce_mean(self.cat_entropy(self.nn.policy_fn))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        params = self.nn._model.trainable_variables()

        print(params)

        pass

    @staticmethod
    def cat_entropy(logits):
        a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keepdims=True)
        p0 = ea0 / z0
        return tf.reudce_sum(p0 * (tf.log(z0) - a0), 1)

    def load(self, load_path):
        self.nn.load_model(load_path)

    def save(self, save_path):
        # Save the weights
        self.nn.save_model(save_path)


if __name__ == '__main__':
    agent = Agent(A2CAgentCNN, VisionShape(8, 8, 1),
                     num_workers=1, num_steps=5)
