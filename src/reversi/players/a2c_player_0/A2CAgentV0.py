import os
import numpy as np
import tensorflow as tf

from reversi.players.a2c_player_0.A2CAgentCNN import VisionShape, A2CAgentCNN

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class A2CAgentV0:
    # is the Agent.
    # Agent can be trained. saves its NN, loads NN etc.
    def __init__(self,
                 nn_class,
                 vision_shape: VisionShape = VisionShape(8, 8, 1),
                 action_size: int = 8*8+1,
                 num_workers: int = 1,
                 num_steps: int = 5,
                 lr=7e-4,
                 ent_coef=0.01, vf_coef=0.5,
                 max_grad_norm=0.5,
                 opt_alpha=0.99, opt_epsilon=1e-5,
                 total_timesteps=int(10e3)):
        # TODO threads
        # tf.config.threading.set_intra_op_parallelism_threads(0)  # let system decide
        # tf.config.threading.set_inter_op_parallelism_threads(0)  # let system decide
        # gpu_devices = tf.config.list_physical_devices('GPU')
        # if len(gpu_devices) > 0:
        # tf.config.experimental.set_memory_growth(gpu_devices[0], True)
        sess = tf.compat.v1.Session()
        self.session = sess

        batch_size = num_workers * num_steps
        self.model = nn_class(batch_size=batch_size,
                              vision_shape=vision_shape,
                              action_size=action_size)

        A = tf.keras.Input(shape=(action_size,), dtype=tf.dtypes.int8)
        ADV = tf.keras.Input(shape=(1,), dtype=tf.dtypes.float32)
        R = tf.keras.Input(shape=(1,), dtype=tf.dtypes.float32)
        LR = tf.keras.Input([], dtype=tf.dtypes.float32)

        # build the loss_fn
        # neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=A, logits=self.model.policy_fn)
        # pg_loss = tf.reduce_mean(ADV * neglogpac)
        weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        # policy gradients, weighted by advantages. only calc on actions we take.
        actions = tf.cast(A, tf.int32)
        policy_gradient_loss = weighted_sparse_ce(
            actions, self.model.policy_fn, sample_weight=ADV)

        vf_loss = tf.reduce_mean(
            tf.math.squared_difference(tf.squeeze(self.model.value_fn),
                                       R) / 2.0)
        entropy = tf.reduce_mean(self.cat_entropy(self.model.policy_fn))
        loss_fn = policy_gradient_loss - entropy * ent_coef + vf_loss * vf_coef

        params = self.model._model.trainable_variables
        grads = tf.gradients(loss_fn, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_params = list(zip(grads, params))
        trainer = tf.optimizers.RMSprop(learning_rate=LR,
                                        decay=opt_alpha,
                                        epsilon=opt_epsilon)
        train_fn = trainer.apply_gradients(grads_and_params)

        def train_a_batch(self, states, rewards, actions, values):
            advs = rewards - values
            feed_dict = {self.model._model.X: states,
                         A: actions,
                         ADV: advs,
                         R: rewards,
                         LR: lr, }
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [policy_gradient_loss, vf_loss, entropy, train_fn],
                feed_dict)
            return policy_loss, value_loss, policy_entropy
        self.train_a_batch = train_a_batch
        self.step = self.model.step

    @ staticmethod
    def cat_entropy(logits):
        a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), 1)

    def load(self, load_path):
        self.model.load_model(load_path)

    def save(self, save_path):
        # Save the weights
        self.model.save_model(save_path)


class A2CAgentV0Trainer:

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    agent = A2CAgentV0(A2CAgentCNN,
                       VisionShape(8, 8, 1),
                       num_workers=1,
                       num_steps=5)
    
    # can we train now? 
    print('legacy version built.')
    agent.session.close()
