import os
import numpy as np
import tensorflow as tf

from reversi.players.a2c_player.A2CAgentCNN import VisionShape, A2CAgentCNN

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class A2CAgent:
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
        pass


if __name__ == '__main__':
    agent = A2CAgent(A2CAgentCNN, VisionShape(8, 8, 1),
                     num_workers=1, num_steps=5)
