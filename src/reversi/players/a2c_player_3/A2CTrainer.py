from typing import Any, List, Sequence, Tuple
import numpy as np
import tensorflow as tf
import tqdm

from reversi.tf_utils import set_global_seeds
from reversi.players.a2c_player_3.A2CModel import A2CModel

import gym


class A2CTrainer:
    # huber_loss: L_d(G, V) - less sensitive to outliers in data than squared-error loss
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    # Small epsilon value for stabilizing division operations
    eps = np.finfo(np.float32).eps.item()

    def __init__(self, env):
        self.env = env  # gym.make("CartPole-v0")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        def _env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Returns state, reward and done flag given an action."""
            # Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
            # This would allow it to be included in a callable TensorFlow graph.
            state, reward, done, _ = self.env.step(action)
            return (state.astype(np.float32),
                    np.array(reward, np.int32),
                    np.array(done, np.int32))

        return tf.numpy_function(_env_step, [action],
                                 [tf.float32, tf.int32, tf.int32])

    def run_episode(self,
                    initial_state: tf.Tensor,
                    model: tf.keras.Model,
                    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, ]:
        """Runs a single episode to collect training data."""

        action_probs = tf.TensorArray(
            dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):
            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)

            # Run the model and to get action probabilities and critic value
            action_logits_t, value = model(state)

            # Sample next action from the action probability distribution
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            # Store critic values
            values = values.write(t, tf.squeeze(value))

            # Store log probability of the action chosen
            action_probs = action_probs.write(t, action_probs_t[0, action])

            # Apply action to the environment to get next state and reward
            state, reward, done = self.tf_env_step(action)
            state.set_shape(initial_state_shape)

            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    @staticmethod
    def get_expected_return(rewards: tf.Tensor,
                            gamma: float,
                            standardize: bool = True) -> tf.Tensor:
        """Compute expected returns per timestep."""

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array. [::-1] reverts an array.
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) /
                       (tf.math.reduce_std(returns) + A2CTrainer.eps))  # eps to avoid /0 error

        return returns

    @staticmethod
    def compute_loss(action_probs: tf.Tensor,
                     values: tf.Tensor,
                     returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = A2CTrainer.huber_loss(values, returns)

        return actor_loss + critic_loss

    @tf.function
    def train_step(self,
                   initial_state: tf.Tensor,
                   model: tf.keras.Model,
                   optimizer: tf.keras.optimizers.Optimizer,
                   gamma: float,
                   max_steps_per_episode: int) -> tf.Tensor:
        """Runs a model training step."""

        # 1. run game for one episode. Collect training data, -> loss
        with tf.GradientTape() as tape:
            # Run the model for one episode to collect training data
            action_probs, values, rewards = self.run_episode(initial_state, model,
                                                             max_steps_per_episode)

            # Calculate expected returns
            returns = A2CTrainer.get_expected_return(rewards, gamma)

            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            # Calculating loss values to update our network
            loss = A2CTrainer.compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, model.trainable_variables)

        # Apply the gradients to the model's parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward

    def train(self,
              model,  # the model to train
              max_episodes=10000,
              max_steps_per_episode=1000,
              gamma=0.99):
        """
        gamma - Discount factor for future rewards
        """

        # Cartpole-v0 is considered solved if average reward is >= 195 over 100
        # consecutive trials
        reward_threshold = 195
        running_reward = 0

        with tqdm.trange(max_episodes) as t:
            for i in t:
                initial_state = tf.constant(self.env.reset(), dtype=tf.float32)
                episode_reward = int(self.train_step(initial_state, model,
                                                     self.optimizer, gamma, max_steps_per_episode))

                running_reward = episode_reward*0.01 + running_reward*.99

                t.set_description(f'Episode {i}')
                t.set_postfix(
                    episode_reward=episode_reward, running_reward=running_reward)

                # Show average episode reward every 10 episodes
                if i % 10 == 0:
                    pass  # print(f'Episode {i}: average reward: {avg_reward}')

                if running_reward > reward_threshold:
                    break

        print(
            f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
