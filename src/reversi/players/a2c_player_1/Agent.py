import numpy as np
import tensorflow as tf


class AgentModel:
    def __init__(self, input_size: int = 8*8, action_size: int = 8*8):

        # create the network
        with tf.name_scope('model'):
            X = tf.keras.Input(shape=(input_size,), dtype=tf.dtypes.float32)
            dense_1 = self._create_dense_layer(512)(X)
            dense_2 = self._create_dense_layer(512)(dense_1)
            dense_3 = self._create_dense_layer(512)(dense_2)
            # fork: to policy and value_fn
            policy_logits = self._create_dense_layer(
                action_size, act_fn=None)(dense_3)
            value_fn = self._create_dense_layer(1, act_fn=None)(dense_3)

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

    @staticmethod
    def _sample(logits):
        noise = tf.random.uniform(tf.shape(logits))
        return tf.argmax(logits - tf.math.log(-tf.math.log(noise)), 1)

    @tf.function
    def step(self, observation):
        output_actions, output_values = self._model.call(observation)
        action = self._sample(output_actions)
        return (action, output_values[:, 0])

        # observation: in shape of: (width, height, depth * time_window_size)
        # return (actions, value) only one output
        # Should ???? observation should be padded to batch size, depth, time_window_size
        # return tf.py_function(_step, [], [tf.int64, tf.float32])
        # return tf.py_function(func=_step, inp=[], Tout=[tf.int32, tf.float32])


class A2CAgent:
    def __init__(self, model,
                 learn_rate=7e-3, ent_coef=0.01, value_coef=0.5, max_grad_norm=0.5,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(10e3)):
        self.value_coef = value_coef
        self.ent_coef = ent_coef

        self.model = model
        self.model._model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learn_rate),
                           loss=[self._logits_loss, self._value_loss])

    def train(self, env, batch_size=10, updates=40):
        # batch arrayes
        actions = np.empty((batch_size,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        observations = np.empty((batch_size, 8*8))

        # training loop
        # ????

    def _value_loss(self, returns, value):
        # build the loss_fn
        return self.value_coef * tf.keras.losses.mean_squared_error(returns, value)

    def _logits_loss(self, actions_and_advantages, logits):
        # split the advantage and actions
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        # policy gradients, weighted by advantages. only calc on actions we take.
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(
            actions, logits, sample_weight=advantages)
        # entropy loss, by cross-entropy over itself
        probs = tf.nn.softmax(logits)
        entropy_loss = tf.keras.losses.categorical_crossentropy(probs, probs)
        # loss function: to minimize policy and maximize entropy losses.
        # flip signs, optimizer minimizes
        return policy_loss - self.ent_coef * entropy_loss


if __name__ == '__main__':
    tf.Tensor
    # create new instance
    agent_model = AgentModel()
    actions, values = agent_model.step(np.zeros((5, 8*8), dtype=np.float32))

    # tf.config.run_functions_eagerly(True)
    # print('run function eagerly:', tf.executing_eagerly())

    print('actions:', actions)
    print('actions:', tf.make_ndarray(tf.make_tensor_proto(actions)))
    print('values:', values)
    print('values:', tf.make_ndarray(tf.make_tensor_proto(values)))

    # create the agent
    agent = A2CAgent(agent_model)

    agent_model._model.summary()
